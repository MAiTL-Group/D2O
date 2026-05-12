import argparse
from PIL import Image
from tqdm import tqdm
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils.tools import set_random_seed

from clip import clip

from data.cls_to_names import get_classnames, CUSTOM_TEMPLATES, ensemble, imagenet_templates
import json

@torch.no_grad()
def pre_extract_class_feature(clip_model, dataset_name, args):
    print("Evaluating: {}".format(dataset_name))
    project_root = os.path.dirname(os.path.abspath(__file__))
    descriptor_root = args.descriptor_path
    if not os.path.isabs(descriptor_root):
        descriptor_root = os.path.join(project_root, descriptor_root)

    if dataset_name == 'imagenet_c':
        classnames = get_classnames('imagenet')
    else:
        classnames = get_classnames(dataset_name, data_root=args.data if args.data else None)

    if args.class_type == "Ensemble":
        template = ensemble
    elif args.class_type == "Img_temp":
        template = imagenet_templates
    elif args.class_type == "Vanilla":
        template = ["a photo of a {}."]
    else:
        template_key = 'imagenet' if dataset_name.startswith('pug') else dataset_name
        template = CUSTOM_TEMPLATES[template_key]
        args.class_type = "Custom"

    if args.GPT:
        save_dir = os.path.join(
            project_root,
            "pre_extracted_class_feat",
            args.arch.replace('/', ''),
            f"GPT_w_{args.class_type}_class_emb",
        )

        if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2', 'imagenet_c'] or dataset_name.startswith('pug'):
            description_file = os.path.join(descriptor_root, 'imagenet.json')
        else:
            description_file = os.path.join(descriptor_root, f'{dataset_name}.json')
        print(f'Using description file: {description_file}')
    else:
        save_dir = os.path.join(
            project_root,
            "pre_extracted_class_feat",
            args.arch.replace('/', ''),
            f"{args.class_type}_class_emb",
        )
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)


    if args.GPT:
        llm_descriptions = json.load(open(description_file))
        clip_weights = []
        for classname in classnames:
            assert len(llm_descriptions[classname]) >= args.num_descriptor
            prompts = [t.format(f"{classname}, " + c).replace("..", ".") for c in llm_descriptions[classname] for t in template]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

            with torch.cuda.amp.autocast():
                class_embeddings = clip_model.encode_text(prompts)  # n_desc x d

                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()

    else:
        clip_weights = []
        for classname in classnames:
            prompts = [tem.format(classname.replace("_", " ")) for tem in template]
            texts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()

    save_name = 'pug_imagenet' if dataset_name.startswith('pug') else dataset_name
    save_path = os.path.join(save_dir, f"{save_name}.pth")
    torch.save(clip_weights, save_path)
    print(f"Successfully save image features to [{save_path}]")

def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.arch)
    clip_model.eval()

    datasets = args.test_set.split('/')
    for dataset_name in datasets:
        print("Extracting features for: {}".format(dataset_name))
        pre_extract_class_feature(clip_model, dataset_name, args)# Pre extract the class embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-extracting image features')
    parser.add_argument('--data', metavar='DIR', default='', help='path to dataset root')
    parser.add_argument('--test_set', type=str, help='dataset name', default='')
    parser.add_argument('--vlm_name', default='CLIP', type=str, help="Type of VLMs")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help=" CLIP model backbone:'RN50' or'ViT-B/16'.")
    parser.add_argument('--seed', type=int, default=0)

    ### ImageNet-c
    parser.add_argument('--level', type=str, default='5', help="Corruption Level")
    parser.add_argument('--corruption', type=str, default='gaussian_noise/shot_noise/impulse_noise/defocus_blur/glass_blur/motion_blur/zoom_blur/snow/frost/fog/brightness/contrast/elastic_transform/pixelate/jpeg_compression', help="corruption type for ImageNet-c")

    ### class embedding
    parser.add_argument('--class_type', default='Custom', type=str, help="Type of the initialization of mean matrix: Custom,Vanilla,Img_temp,Ensemble")
    parser.add_argument('--GPT', action='store_true', help="use the description or not ")
    parser.add_argument('--descriptor_path', type=str, default='./descriptions')
    parser.add_argument('--num_descriptor', type=int, default=50)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)
