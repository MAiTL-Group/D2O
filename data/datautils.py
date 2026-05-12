from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from data.fewshot_datasets import *

# ID_to_DIRNAME={
#     'imagenet': 'domain_shift_datastes/imagenet/images/val',
#     'imagenet_a': 'domain_shift_datastes/imagenet-a/imagenet-a',
#     'imagenet_sketch': 'domain_shift_datastes/ImageNet-Sketch/images',
#     'imagenet_r': 'domain_shift_datastes/imagenet-r/imagenet-r',
#     'imagenetv2': 'domain_shift_datastes/imagenetv2-matched-frequency-format-val/imagenetv2-matched-frequency-format-val',
#     'imagenet_c': 'corruption/imagenet-c',
#     'oxford_flowers': 'fine-grained/oxford_flowers',
#     'dtd': 'fine-grained/dtd',
#     'oxford_pets': 'fine-grained/oxford_pets',
#     'stanford_cars': 'fine-grained/stanford_cars',
#     'ucf101': 'fine-grained/ucf101',
#     'caltech101': 'fine-grained/caltech-101',
#     'food101': 'fine-grained/food-101',
#     'sun397': 'fine-grained/sun397',
#     'fgvc_aircraft': 'fine-grained/fgvc_aircraft',
#     'eurosat': 'fine-grained/eurosat',
# }

ID_to_DIRNAME={
    'imagenet': 'domain_shift_datastes/imagenet/images/val',
    'imagenet_a': 'imagenet-adversarial/imagenet-a',
    'imagenet_sketch': 'imagenet-sketch/ImageNet-Sketch',
    'imagenet_r': 'imagenet-rendition/imagenet-r',
    'imagenetv2': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'imagenet_c': 'imagenet-c',
    'pug_imagenet': 'PUG_ImageNet',
    'oxford_flowers': 'oxford_flowers',
    'dtd': 'dtd',
    'oxford_pets': 'oxford_pets',
    'stanford_cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'fgvc_aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
}

def _format_pug_variant_name(variant: str) -> str:
    parts = [p for p in variant.replace('-', '_').split('_') if p]
    return '_'.join([p[:1].upper() + p[1:] for p in parts])

def _parse_pug_variant(set_id: str) -> str:
    alias = {
        'cpitch': 'Camera_Pitch',
        'croll': 'Camera_Roll',
        'cyaw': 'Camera_Yaw',
        'opitch': 'Object_Pitch',
        'oroll': 'Object_Roll',
        'oyaw': 'Object_Yaw',
        'oscale': 'Object_Scale',
        'otexture': 'Object_Texture',
        'slight': 'Scene_Light',
        'worlds': 'Worlds',
    }

    if set_id == 'pug_imagenet':
        return 'Worlds'
    if set_id == 'pug':
        return 'Worlds'

    if set_id.startswith('pug_imagenet_'):
        suffix = set_id[len('pug_imagenet_'):]
        return _format_pug_variant_name(alias.get(suffix.lower(), suffix))
    if set_id.startswith('pug_'):
        suffix = set_id[len('pug_'):]
        return _format_pug_variant_name(alias.get(suffix.lower(), suffix))

    return 'Worlds'

def build_test_loader(set_id, transform, data_root, batch_size, corruption=None, level=None, num_views=63, shuffle=True, num_workers=8):
    if set_id.startswith('pug_imagenet') or set_id.startswith('pug_') or set_id == 'pug':
        transform = get_ood_preprocess(num_views)
        variant = _parse_pug_variant(set_id)
        testdir = os.path.join(data_root, ID_to_DIRNAME['pug_imagenet'], variant)
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['imagenet_a', 'imagenet_sketch', 'imagenet_r', 'imagenetv2']:
        transform = get_ood_preprocess(num_views)
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id =='imagenet':
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id =='imagenet_c':
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], corruption, level)
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform)
    else:
        raise NotImplementedError
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return val_loader


# Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def aug(image, preprocess):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed


class Augmenter(object):
    def __init__(self, base_transform, preprocess, n_views=63):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        if self.n_views == 0:
            return image
        else:
            views = [aug(x, self.preprocess) for _ in range(self.n_views)]
            return [image] + views

def get_ood_preprocess(num_views):
    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])

    data_transform = Augmenter(base_transform, preprocess, n_views=num_views) # w/O mix_augmentation

    return data_transform
