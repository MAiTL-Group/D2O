import argparse
import time
import os
import math
from datetime import datetime
from typing import Optional, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.tools import Summary, AverageMeter, accuracy, set_random_seed
from data.cls_to_names import custom_scale
from data.datautils import build_test_loader
from clip import clip


def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def _safe_name(name: str) -> str:
    return "".join([c if (c.isalnum() or c in "._-") else "_" for c in str(name)])


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _clip_unnormalize(x):
    return x * _CLIP_STD.to(x.device, x.dtype) + _CLIP_MEAN.to(x.device, x.dtype)


def _clip_normalize(x01):
    return (x01 - _CLIP_MEAN.to(x01.device, x01.dtype)) / _CLIP_STD.to(x01.device, x01.dtype)


def _get_aug_cfg_for_dataset(dataset_name):
    name = (dataset_name or "").lower()
    if "pug" in name:
        return {"brightness": 0.04, "contrast": 0.06, "saturation": 0.06, "rotate": 5.0, "translate_y": 0.05, "scale": 0.06}
    if "imagenet" in name:
        return {"brightness": 0.06, "contrast": 0.08, "saturation": 0.08, "rotate": 3.0, "scale": 0.05}
    return {"brightness": 0.06, "contrast": 0.08, "saturation": 0.08, "rotate": 2.0, "scale": 0.04}


def _adjust_brightness(x01, factor):
    return (x01 * factor).clamp(0.0, 1.0)


def _adjust_contrast(x01, factor):
    mean = x01.mean(dim=(0, 1, 2), keepdim=True)
    return ((x01 - mean) * factor + mean).clamp(0.0, 1.0)


def _adjust_saturation(x01, factor):
    r, g, b = x01[0:1], x01[1:2], x01[2:3]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return ((x01 - gray) * factor + gray).clamp(0.0, 1.0)


def _affine_transform_batch(images01, angle_deg=0.0, translate_px=(0.0, 0.0), scale=1.0):
    n, c, h, w = images01.shape
    device = images01.device
    dtype = images01.dtype

    angle = float(angle_deg) * math.pi / 180.0
    cos_a = math.cos(angle) * float(scale)
    sin_a = math.sin(angle) * float(scale)

    tx_px, ty_px = float(translate_px[0]), float(translate_px[1])
    tx = 2.0 * tx_px / max(1.0, (w - 1.0))
    ty = 2.0 * ty_px / max(1.0, (h - 1.0))

    theta = torch.tensor([[cos_a, -sin_a, tx],
                          [sin_a, cos_a, ty]], device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)

    grid = F.affine_grid(theta, size=images01.size(), align_corners=False)
    out = F.grid_sample(images01, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def _apply_aug_batch(images01, kind, eps, sign):
    kind = kind.lower()
    if kind == "brightness":
        return _adjust_brightness(images01, factor=max(0.0, 1.0 + sign * eps))
    if kind == "contrast":
        outs = [_adjust_contrast(images01[i], factor=max(0.0, 1.0 + sign * eps)) for i in range(images01.size(0))]
        return torch.stack(outs, dim=0)
    if kind == "saturation":
        outs = [_adjust_saturation(images01[i], factor=max(0.0, 1.0 + sign * eps)) for i in range(images01.size(0))]
        return torch.stack(outs, dim=0)
    if kind == "rotate":
        return _affine_transform_batch(images01, angle_deg=sign * eps, translate_px=(0.0, 0.0), scale=1.0)
    if kind == "translate_y":
        _, _, h, _ = images01.shape
        ty = sign * eps * h
        return _affine_transform_batch(images01, angle_deg=0.0, translate_px=(0.0, ty), scale=1.0)
    if kind == "scale":
        return _affine_transform_batch(images01, angle_deg=0.0, translate_px=(0.0, 0.0), scale=max(0.1, 1.0 + sign * eps))
    return images01


@torch.no_grad()
def _encode_normed(encoder, images_normed):
    feats = encoder(images_normed)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float()


@torch.no_grad()
def _finite_diff_direction(images01, encoder, kind, eps):
    x_plus01 = _apply_aug_batch(images01, kind, eps, sign=+1.0)
    x_minus01 = _apply_aug_batch(images01, kind, eps, sign=-1.0)
    z_plus = _encode_normed(encoder, _clip_normalize(x_plus01))
    z_minus = _encode_normed(encoder, _clip_normalize(x_minus01))
    denom = (2.0 * float(eps)) if float(eps) != 0.0 else 1.0
    return (z_plus - z_minus) / denom


@torch.no_grad()
def _collect_domain_images(loader, max_samples, device):
    collected = []
    for images, _ in loader:
        img = images[0] if isinstance(images, list) else images
        if img.dim() == 3:
            img = img.unsqueeze(0)
        collected.append(img[0].to(device=device, non_blocking=True))
        if len(collected) >= max_samples:
            break
    if not collected:
        return None
    return torch.stack(collected, dim=0)


@torch.no_grad()
def estimate_style_basis_U(encoder,
                           loader,
                           device,
                           dataset_name="",
                           max_samples=128,
                           shrinkage=0.1,
                           var_ratio=0.6,
                           max_rank=16,
                           min_rank=1,
                           min_mp_ratio=0.05):
    base = _collect_domain_images(loader, max_samples=max_samples, device=device)
    if base is None:
        return None
    images01 = _clip_unnormalize(base).clamp(0.0, 1.0)

    aug_cfg = _get_aug_cfg_for_dataset(dataset_name)
    deltas = []
    for kind, eps in aug_cfg.items():
        delta = _finite_diff_direction(images01, encoder, kind=kind, eps=float(eps))
        norms = delta.norm(dim=-1)
        med = norms.median()
        delta = delta / (med + 1e-6)
        deltas.append(delta)

    deltas_all = torch.cat(deltas, dim=0).float()
    d = deltas_all.size(1)
    n_tot = deltas_all.size(0)

    C = deltas_all.t() @ deltas_all / float(n_tot)
    trace = torch.trace(C)
    I = torch.eye(d, device=C.device, dtype=C.dtype)
    C = (1.0 - shrinkage) * C + shrinkage * (trace / d) * I

    C_cpu = C.detach().cpu()
    eigvals, eigvecs = torch.linalg.eigh(C_cpu)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    tail_len = max(1, int(d * 0.5))
    noise_var = eigvals[-tail_len:].mean()
    gamma = d / max(1, n_tot)
    lambda_plus = noise_var * (1 + math.sqrt(gamma)) ** 2

    mp_mask = eigvals > lambda_plus
    r_mp = int(mp_mask.sum().item()) if mp_mask.sum() > 0 else 1

    if (r_mp / max(1, max_rank)) < float(min_mp_ratio):
        return None

    total = eigvals.sum()
    cumvar = torch.cumsum(eigvals, dim=0) / (total + 1e-12)
    r_var = int((cumvar < var_ratio).sum().item()) + 1

    r = max(min_rank, min(r_mp, r_var))
    r = min(r, max_rank)

    U = eigvecs[:, :r].to(device=device, dtype=torch.float32)
    return U


@torch.no_grad()
def decompose_content_style(f, U, lambda_proj, style_dim):
    if U is None:
        return f, None
    feat = f.float()
    if style_dim is not None and style_dim > 0 and style_dim < U.size(1):
        U_use = U[:, :style_dim]
    else:
        U_use = U
    z_sty = feat @ U_use
    recon = z_sty @ U_use.t()
    f_cnt = F.normalize(feat - float(lambda_proj) * recon, dim=-1)
    return f_cnt.to(f.dtype), z_sty


def center_logits(s):
    return s - s.mean(dim=-1, keepdim=True)


def init_env_state(max_clusters, num_classes, style_k, device):
    return {
        "max_clusters": int(max_clusters),
        "centers": [],
        "counts": [],
        "bias_logits": [],
        "num_classes": int(num_classes),
        "style_k": int(style_k),
        "device": device,
    }


@torch.no_grad()
def assign_env(z_sty, env_state):
    centers: List[torch.Tensor] = env_state["centers"]
    counts: List[int] = env_state["counts"]
    max_k = env_state["max_clusters"]
    device = env_state["device"]

    z = z_sty[0].detach().to(device=device, dtype=torch.float32)

    if len(centers) < max_k:
        centers.append(z)
        counts.append(1)
        env_state["bias_logits"].append(torch.zeros(env_state["num_classes"], device=device, dtype=torch.float32))
        return len(centers) - 1, 1

    C = torch.stack(centers, dim=0)
    dist2 = ((C - z) ** 2).sum(dim=1)
    idx = int(torch.argmin(dist2).item())
    n = counts[idx]
    centers[idx] = (n * centers[idx] + z) / (n + 1)
    counts[idx] = n + 1
    return idx, counts[idx]


@torch.no_grad()
def update_env_bias(env_idx, s_clip, env_state, rho):
    b_list: List[torch.Tensor] = env_state["bias_logits"]
    if s_clip.dim() == 2:
        s = s_clip[0]
    else:
        s = s_clip
    s = center_logits(s.detach().float())
    b_list[env_idx] = (1.0 - float(rho)) * b_list[env_idx] + float(rho) * s


@torch.no_grad()
def get_env_bias(env_idx, env_state, min_count):
    counts: List[int] = env_state["counts"]
    if env_idx < 0 or env_idx >= len(counts):
        return None, 0
    n = int(counts[env_idx])
    if n < int(min_count):
        return None, n
    return env_state["bias_logits"][env_idx], n


@torch.no_grad()
def param_estimation(image_features, banks, initial_mean, alpha, text_sample_prob):
    """Gaussian distribution parameter estimation with Constructed Knowledge Banks."""

    with torch.no_grad():
        sorted_classes = sorted(banks.keys())

        vecs = torch.cat([item[0].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0)  # (N, feature_dim)
        labels = torch.tensor([class_idx for class_idx in sorted_classes for _ in banks[class_idx]])  # (N)
        cache_pro = torch.cat([item[2].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0)  # (N, num_classes)

        # update mean
        mus = torch.cat([(((cache_pro[labels == i][:, i].unsqueeze(1) * vecs[labels == i]).sum(dim=0) + (image_features * text_sample_prob[:,i].unsqueeze(1)).sum(dim=0)) / ((cache_pro[labels == i][:, i].sum()) + text_sample_prob[:,i].sum())).unsqueeze(0) if i in banks.keys() else initial_mean[i].unsqueeze(0) for i in range(initial_mean.shape[0])])
        mus = alpha * mus + (1 - alpha) * initial_mean

        # KS Estimator (Bayes ridge-type estimator)
        center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in banks.keys()])
        cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())

        ps = torch.ones(initial_mean.shape[0]).cuda() * 1. / initial_mean.shape[0]
        W = torch.einsum('nd, dc -> cn', mus, cov_inv)
        b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

        cache_values = F.one_hot(torch.tensor(labels).to(torch.int64), num_classes=initial_mean.shape[0]).cuda().half()
        return W, b, mus, vecs, labels, cache_pro


@torch.no_grad()
def constructed_knowledge_banks(preds, features_loss, bank_size):
    """
    Update Knowledge Banks by selecting the top 'bank_size' samples for each class based on entropy loss.

    Args:
        cache (dict): Dictionary storing features per class.
        preds (Tensor): Predicted labels for the batch. Shape: (batch_size,)
        features_losses (tuple): (image_features, loss, prob_map), each of shape:
                                 - image_features: (batch_size, feature_dim)
                                 - loss: (batch_size,)
                                 - prob_map: (batch_size, num_classes)
        cbank_size (int): Maximum number of samples to store per class.

    Returns:
        bool: Whether the cache was updated.
    """
    cache = {}
    with torch.no_grad():
        image_features, losses, prob_maps = features_loss
        unique_preds = preds.unique(sorted=True)
        for pred in unique_preds:
            pred = pred.item()
            idxs = (preds == pred).nonzero(as_tuple=True)[0]

            if len(idxs) == 0:
                continue
            if len(idxs) <= bank_size:
                selected_items = [(image_features[i], losses[i].item(), prob_maps[i]) for i in idxs]
            else:
                top_k = losses[idxs].topk(min(len(idxs), bank_size), largest=False)[1]  # top bank_size
                selected_idxs = idxs[top_k]
                selected_items = [(image_features[i], losses[i].item(), prob_maps[i]) for i in selected_idxs]
            cache[pred] = selected_items
    return cache

def process_in_chunk(image_features, clip_weights, chunk_size =64):
    num_samples = image_features.shape[0]
    processed_features, processed_logits = [], []

    for i in range(0, num_samples, chunk_size):
        batch_feat = image_features[i: i + chunk_size]
        batch_logits = 100. * batch_feat.float() @ clip_weights.float()
        batch_entropy = calculate_batch_entropy(batch_logits)
        selected_idx = torch.topk(batch_entropy, max(int(batch_entropy.size(1) * 0.1), 1), dim=1, largest=False).indices
        batch_indices = torch.arange(batch_feat.size(0)).unsqueeze(1).expand_as(selected_idx)
        batch_selected_feat = batch_feat[batch_indices, selected_idx].mean(dim=1)
        batch_selected_logits = batch_logits[batch_indices, selected_idx].mean(dim=1)
        processed_features.append(batch_selected_feat)
        processed_logits.append(batch_selected_logits)
    image_features = torch.cat(processed_features, dim=0)  # [N, D]
    clip_logits = torch.cat(processed_logits, dim=0)  # [N, C]

    return image_features, clip_logits


@torch.no_grad()
def get_clip_logits(image_features, clip_weights):
    if len(image_features.shape)> 2:
        image_features, clip_logits = process_in_chunk(image_features, clip_weights, chunk_size=64)
        loss = calculate_batch_entropy(clip_logits)
        prob_map = clip_logits.softmax(dim=1)
        pred = clip_logits.argmax(dim=1)
    else:
        clip_logits = 100. * image_features.float() @ clip_weights.float()
        loss = calculate_batch_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = clip_logits.argmax(dim=1)
    return image_features, clip_logits, loss, prob_map, pred


@torch.no_grad()
def evaluation(clip_weights, val_loader, clip_model, dataset_name, args):

    # Record evaluation start time and reset peak GPU memory statistics.
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dim = clip_weights.shape[0]
    image_features, target = [], []

    for images, label in tqdm(val_loader):
        label = label.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            if isinstance(images, list):
                # Multi-view data such as PUG and ImageNet variants: images is a
                # list of num_views + 1 tensors, each shaped [B, 3, H, W]. Concatenate
                # them first, then reshape by view count to [B, V, D].
                images_cat = torch.cat(images, dim=0).cuda(non_blocking=True)
                image_fet_flat = clip_model.encode_image(images_cat)
                image_fet_flat = image_fet_flat / image_fet_flat.norm(dim=-1, keepdim=True)
                num_samples = label.size(0)
                num_views = image_fet_flat.size(0) // num_samples
                image_fet = image_fet_flat.view(num_views, num_samples, dim).permute(1, 0, 2)
            else:
                images = images.cuda(non_blocking=True)
                image_fet = clip_model.encode_image(images)
                image_fet = image_fet / image_fet.norm(dim=-1, keepdim=True)
        image_features.append(image_fet)
        target.append(label)

    #==============  Calculate CLIP logits for each image ==============
    image_features = torch.cat(image_features, dim=0)
    target = torch.cat(target, dim=0)
    image_features, clip_logits, _, _, _ = get_clip_logits(image_features, clip_weights)

    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    initial_mean = clip_weights.T
    accuracies = []

    U = None
    env_state = None
    style_dim = args.cpen_style_dim
    if args.cpen:
        device = clip_weights.device
        encoder = clip_model.encode_image
        U = estimate_style_basis_U(
            encoder=encoder,
            loader=val_loader,
            device=device,
            dataset_name=dataset_name,
            max_samples=args.cpen_max_samples,
            shrinkage=args.cpen_shrinkage,
            var_ratio=args.cpen_var_ratio,
            max_rank=args.cpen_max_rank,
            min_mp_ratio=args.cpen_min_mp_ratio,
        )
        if U is not None:
            if style_dim is None or style_dim <= 0:
                style_dim = min(8, U.size(1))
            style_dim = min(style_dim, U.size(1))
            env_state = init_env_state(args.cpen_max_clusters, num_classes=clip_weights.shape[1], style_k=style_dim, device=device)

    if args.cpen and U is not None and env_state is not None:
        logits_prior = clip_logits.clone()
        n_samples = image_features.size(0)
        for i in range(n_samples):
            f_i = image_features[i:i + 1]
            _, z_sty = decompose_content_style(f_i, U, lambda_proj=0.0, style_dim=style_dim)
            z_sty = z_sty / (z_sty.norm(dim=-1, keepdim=True) + 1e-6)
            env_idx, _ = assign_env(z_sty, env_state)
            s_clip = clip_logits[i:i + 1]
            update_env_bias(env_idx, s_clip, env_state, rho=args.env_rho)
            b_e, _ = get_env_bias(env_idx, env_state, min_count=args.cpen_min_cluster_count)
            if b_e is not None and args.gamma_env > 0:
                if args.env_max_abs > 0:
                    mag = b_e.abs().mean().clamp(min=1e-6).item()
                    gamma_eff = min(float(args.gamma_env), float(args.env_max_abs) / float(mag))
                else:
                    gamma_eff = float(args.gamma_env)
                logits_prior[i:i + 1] = s_clip - gamma_eff * b_e.view(1, -1).to(s_clip.device, s_clip.dtype)
            else:
                logits_prior[i:i + 1] = s_clip
        clip_logits_prior = logits_prior
    else:
        clip_logits_prior = clip_logits

    loss = calculate_batch_entropy(clip_logits_prior)
    prob_map = clip_logits_prior.softmax(dim=1)
    preds = clip_logits_prior.argmax(dim=1)

    banks = constructed_knowledge_banks(preds, [image_features, loss, prob_map], args.bank_size)

    W, b, mean, vecs, labels, cache_pro = param_estimation(image_features, banks, initial_mean, args.alpha, text_sample_prob=prob_map)
    GDA_logits = image_features.float() @ W + b

    if args.cpen and U is not None:
        feat_cnt, _ = decompose_content_style(image_features, U, lambda_proj=args.cpen_lambda_proj, style_dim=style_dim)
        vecs_cnt, _ = decompose_content_style(vecs, U, lambda_proj=args.cpen_lambda_proj, style_dim=style_dim)
        similarity_matrix = feat_cnt @ vecs_cnt.T
    else:
        similarity_matrix = image_features @ vecs.T

    if args.sim_relu:
        similarity_matrix = similarity_matrix.relu()
    if args.sim_topk > 0 and similarity_matrix.size(1) > args.sim_topk:
        topk = int(args.sim_topk)
        vals, idx = torch.topk(similarity_matrix, k=topk, dim=1, largest=True)
        pruned = torch.zeros_like(similarity_matrix)
        pruned.scatter_(1, idx, vals)
        similarity_matrix = pruned

    cache_values = F.one_hot(torch.tensor(labels).to(torch.int64), num_classes=initial_mean.shape[0]).to(cache_pro.device).float()
    cache_logits = cache_pro * cache_values

    if args.prior == "prob":
        clip_prior = clip_logits_prior.softmax(dim=1)
    else:
        clip_prior = clip_logits_prior

    test_logits = compute_final_prediction(clip_prior, GDA_logits, similarity_matrix, cache_logits, args)

    acc = accuracy(test_logits, target, topk=(1,))
    top1.update(acc[0], 1)
    accuracies.append(acc[0].item())

    # Summarize total runtime and peak GPU memory.
    end_time = time.time()
    elapsed_time = end_time - start_time
    max_mem_mb = None
    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Elapsed time: {elapsed_time:.2f} seconds, Max GPU memory: {max_mem_mb:.2f} MB")
    else:
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return sum(accuracies) / len(accuracies), elapsed_time, max_mem_mb


def compute_final_prediction(clip_prior, GDA_logits, similarity_matrix, cache_logits, args):
    logP = torch.log_softmax(GDA_logits, dim=1)
    intermediate = logP
    if cache_logits.numel() > 0:
        intermediate += (args.scale / (len(cache_logits) * 2)) * (similarity_matrix @ cache_logits)
    intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
    final_logits = clip_prior * torch.exp((1.0 / args.scale) * intermediate)
    return final_logits


def _open_dataset_log(args, dataset_name: str, run_name: str):
    log_root = os.path.join(args.output_dir, "logs")
    _ensure_dir(log_root)
    fpath = os.path.join(log_root, f"{_safe_name(run_name)}.txt")
    fh = open(fpath, "w", encoding="utf-8")
    args_dict = vars(args)
    fh.write(f"dataset={dataset_name}\nrun_name={run_name}\narch={args.arch}\nseed={args.seed}\n")
    for k in sorted(args_dict.keys()):
        if k == "test_set":
            continue
        fh.write(f"{k}={args_dict[k]}\n")
    fh.write("\n")
    fh.flush()
    return fh, fpath


def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model, preprocess = clip.load(args.arch)
    clip_model.eval()

    datasets = args.test_set.split('/')
    for dataset_name in datasets:
        print("Extracting features for: {}".format(dataset_name))
        if dataset_name.startswith('pug_'):
            args.scale = custom_scale['pug_imagenet']
        else:
            args.scale = custom_scale[dataset_name]
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.arch}_{dataset_name}_{date}"


        #============================clip_weights  ============================
        print("Evaluating: {}".format(dataset_name))
        if args.GPT:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = os.path.join(_project_root(), "pre_extracted_class_feat", args.arch.replace('/', ''), f"GPT_w_{args.class_type}_class_emb")
        else:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = os.path.join(_project_root(), "pre_extracted_class_feat", args.arch.replace('/', ''), f"{args.class_type}_class_emb")

        embed_name = 'pug_imagenet' if dataset_name.startswith('pug') else dataset_name
        clip_weights = torch.load(os.path.join(clip_weights_dir, f"{embed_name}.pth"))

        if dataset_name == 'imagenet_c':
            corruption_type = args.corruption.split('/')
            for corrup in corruption_type:
                run_name = f"{dataset_name}_{corrup}_Transductive"
                log_fh, log_path = _open_dataset_log(args, dataset_name=dataset_name, run_name=run_name)
                print(f"Logging to: {log_path}")
                val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=args.bt, corruption=corrup, level=args.level)

                acc, elapsed_time, max_mem_mb = evaluation(clip_weights, val_loader, clip_model, dataset_name, args)
                log_fh.write(f"final_acc={acc:.4f}\n")
                log_fh.write(f"elapsed_time={elapsed_time:.2f}\n")
                if max_mem_mb is not None:
                    log_fh.write(f"max_gpu_mem_MB={max_mem_mb:.2f}\n")
                log_fh.flush()
                log_fh.close()
        else:
            run_name = f"{dataset_name}_Transductive"
            log_fh, log_path = _open_dataset_log(args, dataset_name=dataset_name, run_name=run_name)
            print(f"Logging to: {log_path}")
            val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=args.bt)

            acc, elapsed_time, max_mem_mb = evaluation(clip_weights, val_loader, clip_model, dataset_name, args)
            log_fh.write(f"final_acc={acc:.4f}\n")
            log_fh.write(f"elapsed_time={elapsed_time:.2f}\n")
            if max_mem_mb is not None:
                log_fh.write(f"max_gpu_mem_MB={max_mem_mb:.2f}\n")
            log_fh.flush()
            log_fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D2O+ADAPT transductive evaluation')
    parser.add_argument('--data', metavar='DIR', default='./datasets/TPT/', help='path to dataset root')
    parser.add_argument('--test_set', type=str, default='imagenet', help='dataset name')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help=" CLIP model backbone:'RN50' or'ViT-B/16'.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bank_size', type=int, default=6, help="The bank size L")
    parser.add_argument('--alpha', type=float, default=0.9, help="the alpha for EMA")

    parser.add_argument('--level', type=str, default='5', help="Corruption Level")
    parser.add_argument('--corruption', type=str, default='gaussian_noise/shot_noise/impulse_noise/defocus_blur/glass_blur/motion_blur/zoom_blur/snow/frost/fog/brightness/contrast/elastic_transform/pixelate/jpeg_compression', help="corruption type for ImageNet-c")

    parser.add_argument('--class_type', default='Custom', type=str, help="Type of the initialization of mean matrix: Custom, Vanilla, Img_temp, Ensemble")
    parser.add_argument('--GPT', action='store_true', default=True, help="use the description or not ")
    parser.add_argument('--bt', type=int, default=64, help="the batch size of test data loader")

    parser.add_argument('--prior', type=str, default='logits', choices=['prob', 'logits'],
                        help='ADAPT prior used in the closed-form fusion')
    parser.add_argument('--view_select_ratio', type=float, default=0.1)

    parser.add_argument('--cpen', action='store_true',
                        help='Enable D2O+ADAPT extensions; flag name kept for backward-compatible scripts')
    parser.add_argument('--cpen_max_samples', type=int, default=128)
    parser.add_argument('--cpen_shrinkage', type=float, default=0.1)
    parser.add_argument('--cpen_var_ratio', type=float, default=0.6)
    parser.add_argument('--cpen_max_rank', type=int, default=16)
    parser.add_argument('--cpen_min_mp_ratio', type=float, default=0.05)
    parser.add_argument('--cpen_style_dim', type=int, default=8)
    parser.add_argument('--cpen_lambda_proj', type=float, default=0.4)

    parser.add_argument('--cpen_max_clusters', type=int, default=16)
    parser.add_argument('--cpen_min_cluster_count', type=int, default=5)
    parser.add_argument('--gamma_env', type=float, default=0.1)
    parser.add_argument('--env_rho', type=float, default=0.05)
    parser.add_argument('--env_max_abs', type=float, default=2.0)

    parser.add_argument('--sim_relu', action='store_true', default=False)
    parser.add_argument('--sim_topk', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs/d2o_adapt_transductive',
                        help='directory for logs')

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)
