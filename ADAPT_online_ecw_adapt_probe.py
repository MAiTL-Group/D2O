
import argparse
import time
import os
import math
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.tools import Summary, AverageMeter, accuracy, set_random_seed
from data.cls_to_names import custom_scale
from clip import clip
from data.datautils import build_test_loader

# ============================================================
# Utilities
# ============================================================

@torch.no_grad()
def calculate_batch_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy for a batch of logits: shape [N, C] -> [N]"""
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

@torch.no_grad()
def avg_entropy(outputs: torch.Tensor) -> torch.Tensor:
    """
    Same as ADAPT: entropy of the average distribution over augmentations.
    outputs: [N, C]
    """
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def _safe_name(name: str) -> str:
    return "".join([c if (c.isalnum() or c in "._-") else "_" for c in str(name)])

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Probe (CSV + optional plots)
# ============================================================

class ProbeWriter:
    def __init__(self, out_dir: str, enabled: bool, plot: bool):
        self.enabled = enabled
        self.plot = plot and enabled
        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, "probe.csv")
        self.summary_path = os.path.join(out_dir, "summary.txt")
        self._rows = []  # keep for plotting/summary
        self._fh = None
        self._header_written = False

        if self.enabled:
            _ensure_dir(out_dir)
            self._fh = open(self.csv_path, "w", encoding="utf-8", newline="")
            # lazy header

    def write_row(self, row: Dict):
        if not self.enabled:
            return
        if (self._fh is None):
            return
        if not self._header_written:
            self._keys = list(row.keys())
            self._fh.write(",".join(self._keys) + "\n")
            self._header_written = True
        values = []
        for k in self._keys:
            v = row.get(k, "")
            if isinstance(v, float):
                values.append(f"{v:.6f}")
            else:
                values.append(str(v))
        self._fh.write(",".join(values) + "\n")
        self._rows.append(row)

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def finalize(self):
        if not self.enabled:
            return
        # summary
        if not self._rows:
            return
        total = len(self._rows)
        clip_correct = sum(int(r.get("correct_clip", 0)) for r in self._rows)
        final_correct = sum(int(r.get("correct_final", 0)) for r in self._rows)
        changed = sum(int(r.get("changed", 0)) for r in self._rows)
        fixed = sum(int(r.get("fixed", 0)) for r in self._rows)
        broken = sum(int(r.get("broken", 0)) for r in self._rows)
        safe_used = sum(int(r.get("safe_used", 0)) for r in self._rows)
        env_applied = sum(int(r.get("env_applied", 0)) for r in self._rows)

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(f"num_samples={total}\n")
            f.write(f"clip_acc={clip_correct/total:.6f}\n")
            f.write(f"final_acc={final_correct/total:.6f}\n")
            f.write(f"changed_rate={changed/total:.6f}\n")
            f.write(f"fixed={fixed}\n")
            f.write(f"broken={broken}\n")
            f.write(f"safe_used_rate={safe_used/total:.6f}\n")
            f.write(f"env_applied_rate={env_applied/total:.6f}\n")

        if self.plot:
            try:
                import matplotlib.pyplot as plt
                # running acc curves
                clip_run = []
                final_run = []
                cc = 0
                fc = 0
                for i, r in enumerate(self._rows, 1):
                    cc += int(r.get("correct_clip", 0))
                    fc += int(r.get("correct_final", 0))
                    clip_run.append(cc / i)
                    final_run.append(fc / i)
                plt.figure()
                plt.plot(clip_run, label="CLIP")
                plt.plot(final_run, label="Final")
                plt.xlabel("step")
                plt.ylabel("running acc")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "acc_curve.png"))
                plt.close()

                # entropy bins
                ents = [float(r.get("ent_norm", 0.0)) for r in self._rows]
                final_ok = [int(r.get("correct_final", 0)) for r in self._rows]
                bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
                bin_acc = []
                bin_mid = []
                for a, b in zip(bins[:-1], bins[1:]):
                    idx = [j for j,e in enumerate(ents) if (e >= a and e < b)]
                    if not idx:
                        bin_acc.append(np.nan)
                    else:
                        bin_acc.append(sum(final_ok[j] for j in idx)/len(idx))
                    bin_mid.append((a+b)/2)
                plt.figure()
                plt.plot(bin_mid, bin_acc, marker="o")
                plt.xlabel("entropy (normalized) bin center")
                plt.ylabel("acc")
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "entropy_bins.png"))
                plt.close()
            except Exception:
                pass

# ============================================================
# CPEN / E-CW-ADAPT core pieces
# ============================================================

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def _clip_unnormalize(x: torch.Tensor) -> torch.Tensor:
    return x * _CLIP_STD.to(x.device, x.dtype) + _CLIP_MEAN.to(x.device, x.dtype)

def _clip_normalize(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - _CLIP_MEAN.to(x01.device, x01.dtype)) / _CLIP_STD.to(x01.device, x01.dtype)

def _get_aug_cfg_for_dataset(dataset_name: str):
    name = (dataset_name or "").lower()
    if "pug" in name:
        return {"brightness": 0.04, "contrast": 0.06, "saturation": 0.06, "rotate": 5.0, "translate_y": 0.05, "scale": 0.06}
    if "imagenet" in name:
        return {"brightness": 0.06, "contrast": 0.08, "saturation": 0.08, "rotate": 3.0, "scale": 0.05}
    return {"brightness": 0.06, "contrast": 0.08, "saturation": 0.08, "rotate": 2.0, "scale": 0.04}

def _adjust_brightness(x01: torch.Tensor, factor: float) -> torch.Tensor:
    return (x01 * factor).clamp(0.0, 1.0)

def _adjust_contrast(x01: torch.Tensor, factor: float) -> torch.Tensor:
    mean = x01.mean(dim=(0, 1, 2), keepdim=True)
    return ((x01 - mean) * factor + mean).clamp(0.0, 1.0)

def _adjust_saturation(x01: torch.Tensor, factor: float) -> torch.Tensor:
    r, g, b = x01[0:1], x01[1:2], x01[2:3]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return ((x01 - gray) * factor + gray).clamp(0.0, 1.0)

def _affine_transform_batch(images01: torch.Tensor,
                            angle_deg: float = 0.0,
                            translate_px=(0.0, 0.0),
                            scale: float = 1.0) -> torch.Tensor:
    N, C, H, W = images01.shape
    device = images01.device
    dtype = images01.dtype

    angle = float(angle_deg) * math.pi / 180.0
    cos_a = math.cos(angle) * float(scale)
    sin_a = math.sin(angle) * float(scale)

    tx_px, ty_px = float(translate_px[0]), float(translate_px[1])
    tx = 2.0 * tx_px / max(1.0, (W - 1.0))
    ty = 2.0 * ty_px / max(1.0, (H - 1.0))

    theta = torch.tensor([[cos_a, -sin_a, tx],
                          [sin_a,  cos_a, ty]], device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)

    grid = F.affine_grid(theta, size=images01.size(), align_corners=False)
    out = F.grid_sample(images01, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.clamp(0.0, 1.0)

@torch.no_grad()
def _apply_aug_batch(images01: torch.Tensor, kind: str, eps: float, sign: float) -> torch.Tensor:
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
        _, _, H, _ = images01.shape
        ty = sign * eps * H
        return _affine_transform_batch(images01, angle_deg=0.0, translate_px=(0.0, ty), scale=1.0)
    if kind == "scale":
        return _affine_transform_batch(images01, angle_deg=0.0, translate_px=(0.0, 0.0), scale=max(0.1, 1.0 + sign * eps))
    return images01

@torch.no_grad()
def _encode_normed(encoder, images_normed: torch.Tensor) -> torch.Tensor:
    feats = encoder(images_normed)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float()

@torch.no_grad()
def _finite_diff_direction(images01: torch.Tensor, encoder, kind: str, eps: float) -> torch.Tensor:
    x_plus01 = _apply_aug_batch(images01, kind, eps, sign=+1.0)
    x_minus01 = _apply_aug_batch(images01, kind, eps, sign=-1.0)
    z_plus = _encode_normed(encoder, _clip_normalize(x_plus01))
    z_minus = _encode_normed(encoder, _clip_normalize(x_minus01))
    denom = (2.0 * float(eps)) if float(eps) != 0.0 else 1.0
    return (z_plus - z_minus) / denom

@torch.no_grad()
def _collect_domain_images(loader, max_samples: int, device: torch.device):
    collected = []
    for (images, _) in loader:
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
                           device: torch.device,
                           dataset_name: str = "",
                           max_samples: int = 128,
                           shrinkage: float = 0.1,
                           var_ratio: float = 0.6,
                           max_rank: int = 16,
                           min_rank: int = 1,
                           min_mp_ratio: float = 0.05):
    """
    Training-free estimate of a low-rank 'style' basis U_sty (d x r).
    This reuses the existing finite-difference orbit approximation (fast + robust).
    Returns U (d x r) or None.
    """
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

    deltas_all = torch.cat(deltas, dim=0).float()  # [M, d]
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

    # Marchenko–Pastur threshold
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
def decompose_content_style(f: torch.Tensor, U: Optional[torch.Tensor], lambda_proj: float, style_dim: int):
    """
    Global, fixed decomposition:
      z_sty = U^T f
      f_cnt = normalize(f - lambda * U U^T f)
    """
    if U is None:
        return f, None
    feat = f.float()
    U_use = U[:, :style_dim] if (style_dim is not None and style_dim > 0 and style_dim < U.size(1)) else U
    z_sty = feat @ U_use  # [B, k]
    recon = z_sty @ U_use.t()
    f_cnt = F.normalize(feat - float(lambda_proj) * recon, dim=-1)
    return f_cnt.to(f.dtype), z_sty

def center_logits(s: torch.Tensor) -> torch.Tensor:
    """cent(s) = s - mean(s) across classes"""
    return s - s.mean(dim=-1, keepdim=True)

# ============================================================
# Environment clusters (style routing) + logits bias (negative cache)
# ============================================================

def init_env_state(max_clusters: int, num_classes: int, style_k: int, device: torch.device):
    return {
        "max_clusters": int(max_clusters),
        "centers": [],          # List[Tensor(k)]
        "counts": [],           # List[int]
        "bias_logits": [],      # List[Tensor(C)] centered logits EMA
        "num_classes": int(num_classes),
        "style_k": int(style_k),
        "device": device
    }

@torch.no_grad()
def assign_env(z_sty: torch.Tensor, env_state: dict) -> Tuple[int, int]:
    """
    Assign a single sample (z_sty[0]) to an env cluster.
    Returns (env_idx, count_after_update_center)
    """
    centers: List[torch.Tensor] = env_state["centers"]
    counts: List[int] = env_state["counts"]
    max_k = env_state["max_clusters"]
    device = env_state["device"]

    z = z_sty[0].detach().to(device=device, dtype=torch.float32)

    if len(centers) < max_k:
        centers.append(z)
        counts.append(1)
        # bias initialized later on first update
        env_state["bias_logits"].append(torch.zeros(env_state["num_classes"], device=device, dtype=torch.float32))
        return len(centers) - 1, 1

    C = torch.stack(centers, dim=0)  # [K, k]
    dist2 = ((C - z) ** 2).sum(dim=1)
    idx = int(torch.argmin(dist2).item())
    n = counts[idx]
    centers[idx] = (n * centers[idx] + z) / (n + 1)
    counts[idx] = n + 1
    return idx, counts[idx]

@torch.no_grad()
def update_env_bias(env_idx: int, s_clip: torch.Tensor, env_state: dict, rho: float):
    """
    Update b_e with EMA of centered CLIP logits (training-free).
    s_clip: [1, C] or [C]
    """
    b_list: List[torch.Tensor] = env_state["bias_logits"]
    if s_clip.dim() == 2:
        s = s_clip[0]
    else:
        s = s_clip
    s = center_logits(s.detach().float())
    b_list[env_idx] = (1.0 - float(rho)) * b_list[env_idx] + float(rho) * s

@torch.no_grad()
def get_env_bias(env_idx: int, env_state: dict, min_count: int) -> Tuple[Optional[torch.Tensor], int]:
    """
    Returns (b_e, count) if count >= min_count else (None, count)
    """
    counts: List[int] = env_state["counts"]
    if env_idx < 0 or env_idx >= len(counts):
        return None, 0
    n = int(counts[env_idx])
    if n < int(min_count):
        return None, n
    return env_state["bias_logits"][env_idx], n

# ============================================================
# ADAPT core: parameter estimation (kept same)
# ============================================================

@torch.no_grad()
def param_estimation(added_sample, banks, initial_mean, prev_mus, alpha):
    """Online Gaussian distribution parameter estimation with Constructed Knowledge Banks."""
    image_features, pred, img_pro = added_sample
    vecs, labels, cache_pro = banks
    cache_keys = torch.unique(labels)

    mus = prev_mus.clone()
    mask = labels == pred
    selected_vecs = vecs[mask]
    selected_cache_pro = cache_pro[mask, pred].unsqueeze(1)

    new_mu = ((selected_cache_pro * selected_vecs).sum(dim=0) + img_pro[0][pred] * image_features[0]) / (
        selected_cache_pro.sum() + img_pro[0][pred]
    ).unsqueeze(0)
    new_mu = alpha * new_mu + (1 - alpha) * initial_mean[pred]
    mus[pred] = new_mu

    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in cache_keys])
    n, d = center_vecs.shape
    if n == 1:
        Sigma = torch.eye(d).to(vecs.device)
    else:
        Sigma = center_vecs.T.cov()
    trace = Sigma.trace()
    cov_inv = d * torch.linalg.pinv((n - 1) * Sigma + trace * torch.eye(d).to(vecs.device))

    ps = torch.ones(initial_mean.shape[0]).to(vecs.device) * 1.0 / initial_mean.shape[0]
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
    return W, b, mus

def update_knowledge_banks_dual(banks, features_loss, bank_size):
    """
    Keep the exact same replacement rule as ADAPT, but write both raw and content keys,
    and store prob_map from the *debiased prior*.
    """
    pred, feat_raw, feat_cnt, loss, prob_map = features_loss
    cache_vecs_raw, cache_vecs_cnt, cache_labels, cache_pro, cache_loss = banks

    start_idx = pred * bank_size
    end_idx = start_idx + bank_size
    existing_count = (cache_labels[start_idx:end_idx] == pred).sum().item()

    update = False
    insert_idx = None
    if existing_count < bank_size:
        insert_idx = start_idx + existing_count
        update = True
    else:
        max_loss_value, max_loss_idx = cache_loss[start_idx:end_idx].max(dim=0)
        if float(loss) < float(max_loss_value.item()):
            insert_idx = start_idx + int(max_loss_idx.item())
            update = True

    if update and (insert_idx is not None):
        cache_vecs_raw[insert_idx] = feat_raw
        cache_vecs_cnt[insert_idx] = feat_cnt
        cache_labels[insert_idx] = pred
        cache_pro[insert_idx] = prob_map
        cache_loss[insert_idx] = loss

    added_sample = [feat_raw, pred, prob_map]
    return update, [cache_vecs_raw, cache_vecs_cnt, cache_labels, cache_pro, cache_loss], added_sample

# ============================================================
# Closed-form prediction (ADAPT, unchanged)
# ============================================================

def compute_final_prediction(clip_prior, GDA_logits, similarity_matrix, cache_logits, args):
    """
    Keep ADAPT structure.
    Note: clip_prior can be prob or logits depending on args.prior. (prob recommended for E-CW-ADAPT)
    """
    logP = torch.log_softmax(GDA_logits, dim=1)
    inter = logP
    if cache_logits.numel() > 0:
        inter += (args.scale / (len(cache_logits) * 2)) * (similarity_matrix @ cache_logits)
    inter -= torch.max(inter, dim=1, keepdim=True)[0]
    return clip_prior * torch.exp((1.0 / args.scale) * inter)

# ============================================================
# Main evaluation loop (E-CW-ADAPT)
# ============================================================

@torch.no_grad()
def _forward_multiview(images, clip_weights, encoder):
    """
    Encodes possibly multi-view input, returning:
      feats_all: [V, d] (normalized)
      logits_all: [V, C] (raw CLIP logits)
    """
    if isinstance(images, list):
        x = torch.cat(images, dim=0).to(clip_weights.device, non_blocking=True)
    else:
        x = images.to(clip_weights.device, non_blocking=True)

    feats = encoder(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    logits = 100.0 * feats.float() @ clip_weights.float()
    return feats.float(), logits


@torch.no_grad()
def _select_low_entropy(feats_all: torch.Tensor,
                        logits_all: torch.Tensor,
                        select_ratio: float = 0.1):
    """
    ADAPT selection: pick lowest entropy views and average.

    Returns:
      feat:      [1, d]
      logits:    [1, C]
      prob_map:  [1, C]
      loss:      float (for bank replacement)
      pred:      int
    """
    if feats_all.size(0) <= 1:
        feat = feats_all
        logits = logits_all
        prob_map = logits.softmax(1)
        loss = float(calculate_batch_entropy(logits).mean().item())
        pred = int(logits.topk(1, 1, True, True)[1].t()[0])
        return feat, logits, prob_map, loss, pred

    ent = calculate_batch_entropy(logits_all)  # [V]
    k = max(int(ent.size(0) * float(select_ratio)), 1)
    selected_idx = torch.topk(ent, k, largest=False).indices
    out = logits_all[selected_idx]                        # [k, C]
    feat = feats_all[selected_idx].mean(0, keepdim=True)  # [1, d]
    logits = out.mean(0, keepdim=True)                    # [1, C]
    loss = float(avg_entropy(out).item())
    prob_map = out.softmax(1).mean(0, keepdim=True)
    pred = int(logits.topk(1, 1, True, True)[1].t()[0])
    return feat, logits, prob_map, loss, pred


@torch.no_grad()
def evaluation(val_loader, clip_weights, image_encoder, args, dataset_name: str, run_name: str, log_fh=None):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    cls_num, dim = clip_weights.shape[1], clip_weights.shape[0]
    initial_mean = clip_weights.T.float()
    mean = None
    W = b = None
    use_raw_wij = bool(getattr(args, "raw_wij", False) or dataset_name == "imagenet_c")
    feature_msg = "[E-CW-ADAPT] similarity_feature=raw" if use_raw_wij else "[E-CW-ADAPT] similarity_feature=content"
    if dataset_name == "imagenet_c" and not getattr(args, "raw_wij", False):
        feature_msg += " (ImageNet-C preset)"
    print(feature_msg)
    if log_fh is not None:
        log_fh.write(feature_msg + "\n")
        log_fh.flush()

    # ---------------- Probe ----------------
    probe = None
    if args.probe:
        probe_root = os.path.join(args.output_dir, "probes", _safe_name(run_name))
        probe = ProbeWriter(probe_root, enabled=True, plot=args.probe_plot)

    # ---------------- Style basis U_sty ----------------
    U = None
    env_state = None
    style_dim = args.cpen_style_dim
    if args.cpen:
        U = estimate_style_basis_U(
            encoder=image_encoder,
            loader=val_loader,
            device=clip_weights.device,
            dataset_name=dataset_name,
            max_samples=args.cpen_max_samples,
            shrinkage=args.cpen_shrinkage,
            var_ratio=args.cpen_var_ratio,
            max_rank=args.cpen_max_rank,
            min_mp_ratio=args.cpen_min_mp_ratio,
        )
        if U is None:
            print("[E-CW-ADAPT] U_sty is None -> fallback to plain ADAPT (no env/style routing, no content weighting).")
        else:
            if style_dim is None or style_dim <= 0:
                style_dim = min(8, U.size(1))
            style_dim = min(style_dim, U.size(1))
            print(f"[E-CW-ADAPT] U rank={U.size(1)}, use style_dim={style_dim}")
            env_state = init_env_state(args.cpen_max_clusters, num_classes=cls_num, style_k=style_dim, device=clip_weights.device)

    # ---------------- Knowledge Banks ----------------
    cache_vecs_raw = torch.zeros((cls_num * args.bank_size, dim), device=clip_weights.device)
    cache_vecs_cnt = torch.zeros((cls_num * args.bank_size, dim), device=clip_weights.device)
    cache_labels = torch.full((cls_num * args.bank_size,), -1, dtype=torch.long, device=clip_weights.device)
    cache_pro = torch.zeros((cls_num * args.bank_size, cls_num), device=clip_weights.device)
    cache_loss = torch.full((cls_num * args.bank_size,), float('inf'), device=clip_weights.device)
    cache = [cache_vecs_raw, cache_vecs_cnt, cache_labels, cache_pro, cache_loss]

    accuracies = []
    clip_accuracies = []
    if clip_weights.is_cuda:
        torch.cuda.reset_peak_memory_stats(clip_weights.device)
    start_time = time.time()

    for step, (images, target) in enumerate(tqdm(val_loader, desc='Processed test images: '), start=1):
        target = target.to(clip_weights.device, non_blocking=True)

        # 1) CLIP forward (multi-view)
        feats_all, logits_raw_all = _forward_multiview(images, clip_weights, image_encoder)

        # 2) Style routing + env bias update (training-free)
        env_idx = -1
        env_count = 0
        b_e = None
        env_applied = 0
        gamma_eff = 0.0

        if args.cpen and (U is not None) and (env_state is not None):
            # use mean feature across views for stable routing
            feat_mean = feats_all.mean(0, keepdim=True)
            _, z_sty = decompose_content_style(feat_mean, U, lambda_proj=0.0, style_dim=style_dim)  # lambda=0: just project
            z_sty = z_sty / (z_sty.norm(dim=-1, keepdim=True) + 1e-6)

            env_idx, env_count = assign_env(z_sty, env_state)

            # update bias using mean raw logits (centered)
            s_clip_mean = logits_raw_all.mean(0, keepdim=True)
            update_env_bias(env_idx, s_clip_mean, env_state, rho=args.env_rho)

            b_e, env_count = get_env_bias(env_idx, env_state, min_count=args.cpen_min_cluster_count)

        # 3) Build debiased prior logits for all views
        logits_prior_all = logits_raw_all
        if (b_e is not None) and args.gamma_env > 0:
            # auto-limit mean |gamma*bias| if asked
            if args.env_max_abs > 0:
                mag = b_e.abs().mean().clamp(min=1e-6).item()
                gamma_eff = min(float(args.gamma_env), float(args.env_max_abs) / float(mag))
            else:
                gamma_eff = float(args.gamma_env)
            logits_prior_all = logits_raw_all - gamma_eff * b_e.view(1, -1).to(logits_raw_all.device, logits_raw_all.dtype)
            env_applied = 1

        # 4) ADAPT selection / confidence uses the debiased prior logits (E-CW-ADAPT Replace #1)
        f_raw, clip_logits_prior, prob_map, loss, pred = _select_low_entropy(feats_all, logits_prior_all, select_ratio=args.view_select_ratio)

        # also keep clip-only (for probe & safe fuse)
        _, clip_logits_raw_sel, _, _, _ = _select_low_entropy(feats_all, logits_raw_all, select_ratio=args.view_select_ratio)
        clip_prob_raw = clip_logits_raw_sel.softmax(1)
        clip_pred_raw = int(clip_logits_raw_sel.topk(1, 1, True, True)[1].t()[0])

        # 5) Feature for similarity (ImageNet-C uses raw features per the final online preset)
        if args.cpen and (U is not None) and (not use_raw_wij):
            f_cnt, _ = decompose_content_style(f_raw, U, lambda_proj=args.cpen_lambda_proj, style_dim=style_dim)
        else:
            f_cnt = f_raw

        # choose clip prior used in closed-form output
        clip_prob_prior = clip_logits_prior.softmax(1)
        clip_prior = clip_prob_prior if args.prior == "prob" else clip_logits_prior

        # 6) Update knowledge bank using debiased prior prob_map/loss/pred
        update_sign, cache, added_sample = update_knowledge_banks_dual(
            cache,
            [pred, f_raw, f_cnt, loss, prob_map],
            args.bank_size,
        )

        cache_vecs_raw, cache_vecs_cnt, cache_labels, cache_pro, cache_loss = cache
        valid_mask = cache_labels != -1
        vecs_raw = cache_vecs_raw[valid_mask]
        vecs_cnt = cache_vecs_cnt[valid_mask]
        labels = cache_labels[valid_mask]
        cache_pro_valid = cache_pro[valid_mask]

        if mean is None:
            mean = initial_mean.clone()

        # 7) Update GDA params (raw space, per ADAPT)
        if update_sign and vecs_raw.numel() > 0:
            W, b, mean = param_estimation(added_sample, [vecs_raw, labels, cache_pro_valid], initial_mean, prev_mus=mean, alpha=args.alpha)

        # 8) Closed-form prediction (with safe fuse option)
        safe_used = 0
        if args.safe_fuse:
            # confidence computed on raw CLIP (more stable against debias errors)
            ent = float(calculate_batch_entropy(clip_logits_raw_sel).item())
            ent_norm = ent / (math.log(cls_num) + 1e-12)
            probs = clip_prob_raw[0]
            top2 = torch.topk(probs, k=2).values
            margin = float((top2[0] - top2[1]).item())
            if (step <= int(args.warmup_steps)) or (ent_norm <= float(args.safe_ent)) or (margin >= float(args.safe_margin)):
                test_logits = (clip_prob_raw if args.prior == 'prob' else clip_logits_raw_sel)
                safe_used = 1
            else:
                test_logits = None
        else:
            test_logits = None

        if test_logits is None:
            if (W is None) or (b is None) or (vecs_raw.numel() == 0):
                test_logits = clip_prior
            else:
                # GDA uses raw feature space (S0)
                GDA_logits = f_raw @ W + b

                # content-weighted similarity (S1); ImageNet-C uses raw features in the online preset
                sim = (f_raw @ vecs_raw.t()) if use_raw_wij else (f_cnt @ vecs_cnt.t())
                if args.sim_relu:
                    sim = sim.relu()

                # cache logits from stored prior probabilities
                cache_values = F.one_hot(labels.to(torch.int64), num_classes=cls_num).to(cache_pro_valid.device).float()
                cache_logits = cache_pro_valid * cache_values

                # optional top-k pruning for stability
                if args.sim_topk > 0 and sim.size(1) > args.sim_topk:
                    topk = int(args.sim_topk)
                    vals, idx = torch.topk(sim, k=topk, dim=1, largest=True)
                    pruned = torch.zeros_like(sim)
                    pruned.scatter_(1, idx, vals)
                    sim = pruned

                test_logits = compute_final_prediction(clip_prior, GDA_logits, sim, cache_logits, args)

        # 9) Metrics
        acc = accuracy(test_logits, target, topk=(1,))
        top1.update(acc[0], 1)
        accuracies.append(float(acc[0].item()))

        # for probe: compute clip-only acc
        clip_acc = accuracy(clip_logits_raw_sel, target, topk=(1,))
        clip_accuracies.append(float(clip_acc[0].item()))

        if probe is not None:
            correct_clip = int(clip_pred_raw == int(target.item()))
            pred_final = int(test_logits.topk(1, 1, True, True)[1].t()[0])
            correct_final = int(pred_final == int(target.item()))
            changed = int(pred_final != clip_pred_raw)
            fixed = int((correct_clip == 0) and (correct_final == 1))
            broken = int((correct_clip == 1) and (correct_final == 0))

            ent_prior = float(calculate_batch_entropy(clip_logits_prior).item())
            ent_norm = ent_prior / (math.log(cls_num) + 1e-12)

            # similarity stats
            sim_top1 = float(sim.max().item()) if ("sim" in locals() and sim.numel() > 0) else 0.0

            probe.write_row({
                "step": step,
                "target": int(target.item()),
                "pred_clip": int(clip_pred_raw),
                "pred_final": int(pred_final),
                "correct_clip": correct_clip,
                "correct_final": correct_final,
                "changed": changed,
                "fixed": fixed,
                "broken": broken,
                "ent_norm": float(ent_norm),
                "bank_n": int(labels.numel()),
                "sim_top1": float(sim_top1),
                "env_applied": int(env_applied),
                "env_count": int(env_count),
                "gamma_eff": float(gamma_eff),
                "safe_used": int(safe_used),
                "raw_wij": int(use_raw_wij),
                "update_bank": int(update_sign),
            })

        if (step == 1) or (step % int(args.log_interval) == 0):
            avg_acc = float(sum(accuracies) / len(accuracies))
            msg = f"[{run_name}] step={step} acc={float(acc[0]):.4f} avg_acc={avg_acc:.4f}"
            print(msg)
            if log_fh is not None:
                log_fh.write(msg + "\n")
                log_fh.flush()

    elapsed_time = time.time() - start_time
    max_mem_mb = None
    if clip_weights.is_cuda:
        max_mem_mb = torch.cuda.max_memory_allocated(clip_weights.device) / (1024 * 1024)
    final_acc = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0
    clip_final_acc = float(sum(clip_accuracies) / len(clip_accuracies)) if clip_accuracies else 0.0
    if max_mem_mb is not None:
        summary_msg = f"[{run_name}] Elapsed time: {elapsed_time:.2f}s, max_gpu_mem={max_mem_mb:.2f}MB, final_acc={final_acc:.4f}, clip_acc={clip_final_acc:.4f}"
    else:
        summary_msg = f"[{run_name}] Elapsed time: {elapsed_time:.2f}s, final_acc={final_acc:.4f}, clip_acc={clip_final_acc:.4f}"
    print(summary_msg)
    if log_fh is not None:
        log_fh.write(summary_msg + "\n")
        log_fh.flush()

    if probe is not None:
        probe.close()
        probe.finalize()

    return final_acc

def _open_dataset_log(args, dataset_name: str, run_name: str):
    log_root = os.path.join(args.output_dir, "txt_logs")
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
    print(f"=> Model created: visual backbone {args.arch}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(args.arch, device=device)
    clip_model.eval()

    datasets = args.test_set.split('/')
    for dataset_name in datasets:
        if dataset_name.startswith('pug_'):
            args.scale = custom_scale['pug_imagenet']
        else:
            args.scale = custom_scale[dataset_name]

        date = datetime.now().strftime("%b%d_%H-%M-%S")
        run_name = f"{dataset_name}_Online_{date}"

        project_root = _project_root()
        if args.GPT:
            clip_weights_dir = os.path.join(project_root, "pre_extracted_class_feat", args.arch.replace('/', ''), f"GPT_w_{args.class_type}_class_emb")
        else:
            clip_weights_dir = os.path.join(project_root, "pre_extracted_class_feat", args.arch.replace('/', ''), f"{args.class_type}_class_emb")

        embed_name = 'pug_imagenet' if dataset_name.startswith('pug') else dataset_name
        clip_weights_path = os.path.join(clip_weights_dir, f"{embed_name}.pth")
        if not os.path.exists(clip_weights_path):
            raise FileNotFoundError(
                f"Missing class embedding file: {clip_weights_path}. "
                f"Please run Pre_extract_class_emb_default.py to generate it first."
            )
        clip_weights = torch.load(clip_weights_path, map_location="cpu").to(device)

        if dataset_name == 'imagenet_c':
            corruption_type = args.corruption.split('/')
            for corrup in corruption_type:
                run_name_c = f"{dataset_name}_{corrup}_Online_{date}"
                log_fh, log_path = _open_dataset_log(args, dataset_name=dataset_name, run_name=run_name_c)
                print(f"Logging to: {log_path}")
                val_loader = build_test_loader(
                    dataset_name, preprocess, args.data,
                    batch_size=1, corruption=corrup, level=args.level,
                    num_views=args.num_views, shuffle=not args.no_shuffle,
                    num_workers=args.num_workers
                )
                try:
                    evaluation(val_loader, clip_weights, clip_model.encode_image, args, dataset_name=dataset_name, run_name=run_name_c, log_fh=log_fh)
                finally:
                    log_fh.close()
        else:
            log_fh, log_path = _open_dataset_log(args, dataset_name=dataset_name, run_name=run_name)
            print(f"Logging to: {log_path}")
            val_loader = build_test_loader(
                dataset_name, preprocess, args.data,
                batch_size=1, num_views=args.num_views,
                shuffle=not args.no_shuffle, num_workers=args.num_workers
            )
            try:
                evaluation(val_loader, clip_weights, clip_model.encode_image, args, dataset_name=dataset_name, run_name=run_name, log_fh=log_fh)
            finally:
                log_fh.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E-CW-ADAPT: Environment-debiased & Content-Weighted ADAPT (training-free)')

    parser.add_argument('--data', metavar='DIR', default='./datasets/TPT/', help='path to dataset root')
    parser.add_argument('--test_set', type=str, default='imagenet', help='dataset name(s), separated by "/"')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help="CLIP model backbone: 'RN50' or 'ViT-B/16'.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bank_size', type=int, default=16, help="Bank Size L")
    parser.add_argument('--alpha', type=float, default=0.9, help="the alpha for EMA")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_views', type=int, default=63)
    parser.add_argument('--no_shuffle', action='store_true', default=False)

    # ImageNet-c
    parser.add_argument('--level', type=str, default='5', help="Corruption Level")
    parser.add_argument('--corruption', type=str, default='gaussian_noise/shot_noise/impulse_noise/defocus_blur/glass_blur/motion_blur/zoom_blur/snow/frost/fog/brightness/contrast/elastic_transform/pixelate/jpeg_compression',
                        help="corruption type for ImageNet-c")

    # class embedding
    parser.add_argument('--class_type', default='Custom', type=str, help="Custom, Vanilla, Img_temp, Ensemble")
    parser.add_argument('--GPT', action='store_true', default=True, help="use the description or not ")

    # ADAPT prior in the closed-form (prob recommended; keep your current default via sh if needed)
    parser.add_argument('--prior', type=str, default='logits', choices=['prob', 'logits'])

    # View selection (same as ADAPT logic; you can keep default)
    parser.add_argument('--view_select_ratio', type=float, default=0.1, help='ratio of low-entropy views to average (ADAPT default ~0.1)')

    # E-CW-ADAPT switches
    parser.add_argument('--cpen', action='store_true', help='Enable E-CW-ADAPT extensions (style routing + content weighting)')
    parser.add_argument('--cpen_max_samples', type=int, default=128, help='Samples to estimate style basis U')
    parser.add_argument('--cpen_shrinkage', type=float, default=0.1)
    parser.add_argument('--cpen_var_ratio', type=float, default=0.6)
    parser.add_argument('--cpen_max_rank', type=int, default=16)
    parser.add_argument('--cpen_min_mp_ratio', type=float, default=0.05, help='If MP-selected rank is too small, skip CPEN (return None)')
    parser.add_argument('--cpen_style_dim', type=int, default=8)
    parser.add_argument('--cpen_lambda_proj', type=float, default=0.4, help='content projection removal strength λ_proj')

    # env bias (logits-domain EMA)
    parser.add_argument('--cpen_max_clusters', type=int, default=16)
    parser.add_argument('--cpen_min_cluster_count', type=int, default=5)
    parser.add_argument('--gamma_env', type=float, default=0.1, help='environment debias strength γ')
    parser.add_argument('--env_rho', type=float, default=0.05, help='EMA rate ρ for logits bias')
    parser.add_argument('--env_max_abs', type=float, default=2.0, help='auto-limit mean |gamma*bias|; <=0 disables')

    # dataset-specific feature mode / ablation support
    parser.add_argument('--raw_wij', action='store_true', default=False,
                        help='Use raw features for similarity weights and cache keys. Online ImageNet-C enables this automatically.')

    # similarity options
    parser.add_argument('--sim_relu', action='store_true', default=False, help='apply ReLU to similarity weights')
    parser.add_argument('--sim_topk', type=int, default=0, help='if >0, keep only top-k similarities (stability)')

    # safety fuse (optional, recommended for hard geometric shifts)
    parser.add_argument('--safe_fuse', action='store_true', default=False, help='enable safe fusion: warmup + CLIP confidence protect')
    parser.add_argument('--warmup_steps', type=int, default=0, help='if safe_fuse, output CLIP prior for first N steps')
    parser.add_argument('--safe_ent', type=float, default=0.20, help='if safe_fuse, output CLIP prior when ent_norm <= safe_ent')
    parser.add_argument('--safe_margin', type=float, default=0.65, help='if safe_fuse, output CLIP prior when prob margin >= safe_margin')

    # probe
    parser.add_argument('--probe', action='store_true', default=False, help='write probe.csv + summary.txt')
    parser.add_argument('--probe_plot', action='store_true', default=False, help='also save probe plots (requires matplotlib)')

    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='./outputs/d2o_adapt_online',
                        help='directory for logs and optional probe files')

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)
