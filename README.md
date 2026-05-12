# D2O

Official code release for **D2O: A Dual Debiasing Operator for Training-Free Test-Time Adaptation of Vision-Language Models**.

D2O is a strictly training-free inference-time operator for CLIP-style vision-language models. For each test sample, it builds:

- `fcnt`: a retrieval-oriented feature with nuisance-sensitive directions suppressed.
- `zsty`: a low-dimensional routing coordinate for environment-aware bias tracking.
- `sdeb`: debiased logits obtained by subtracting cluster-wise centered logit bias.

This repository contains reproducible code for the strongest host-adapter combination used in the paper:

- **D2O+ADAPT**: online and transductive Gaussian-posterior adaptation.

Generated results, pre-extracted features, probe CSVs, checkpoints, and run logs are intentionally excluded. They can be regenerated from the code.

## Repository Layout

```text
.
├── ADAPT_online_ecw_adapt_probe.py        # D2O+ADAPT, online setting
├── ADAPT_transductive_ecw_adapt_probe.py  # D2O+ADAPT, transductive setting
├── Pre_extract_class_emb_default.py        # CLIP text/class embedding pre-extraction
├── clip/                                   # CLIP implementation used by ADAPT host
├── configs/
│   └── d2o_paper_hparams.yaml              # dataset-specific paper preset groups
├── data/                                   # dataset loaders and class-name utilities
├── descriptions/                           # GPT-assisted class descriptions
├── scripts/
│   ├── pre_extract_class_emb.sh
│   ├── run_d2o_adapt_online.sh
│   ├── run_d2o_adapt_online_paper_presets.sh
│   ├── run_d2o_adapt_transductive.sh
│   └── run_d2o_adapt_transductive_paper_presets.sh
├── utils/                                  # shared utilities
├── requirements_adapt.txt                  # ADAPT/D2O+ADAPT environment deps
```

## Environments

D2O+ADAPT follows the ADAPT environment:

```bash
conda create -n d2o_adapt python=3.10
conda activate d2o_adapt
pip install -r requirements_adapt.txt
```

## Data

Set `DATA_ROOT` to the root folder containing the benchmark datasets. The ADAPT-side loader expects names such as:

```text
DATA_ROOT/
├── imagenet-adversarial/imagenet-a
├── imagenet-rendition/imagenet-r
├── imagenet-sketch/ImageNet-Sketch
├── imagenetv2/imagenetv2-matched-frequency-format-val
├── imagenet-c/<corruption>/<level>
├── PUG_ImageNet/<variant>
├── dtd
├── oxford_flowers
└── ...
```

For the fine-grained datasets, keep the CoOp/TPT split JSON files inside each dataset folder, for example `split_zhou_OxfordPets.json`.

For ImageNetV2, use the WordNet-folder layout under `imagenetv2/`:

```text
DATA_ROOT/
└── imagenetv2/
    ├── classnames.txt
    └── imagenetv2-matched-frequency-format-val/
        ├── n01440764/
        ├── n01443537/
        └── ...
```

For ImageNet variants and PUG, also keep the corresponding `classnames.txt` file in the dataset family folder, for example `imagenet-adversarial/classnames.txt`, `imagenetv2/classnames.txt`, and `PUG_ImageNet/classnames.txt`.

## Pre-Extract Class Embeddings

D2O+ADAPT uses CLIP text/class embeddings stored under `pre_extracted_class_feat/`. They are generated, not committed. Generate them once for the `TESTSETS` you plan to evaluate before running the ADAPT preset scripts.

```bash
DATA_ROOT=/path/to/data \
TESTSETS=imagenetv2/imagenet_a/imagenet_r/imagenet_sketch \
MODEL_ARCH=ViT-B/16 \
bash scripts/pre_extract_class_emb.sh
```

For PUG, the script stores the shared embedding as `pug_imagenet.pth`:

```bash
DATA_ROOT=/path/to/data TESTSETS=pug_cpitch bash scripts/pre_extract_class_emb.sh
```

For the full ADAPT paper preset suite, pass the same slash-separated dataset names used by the preset scripts, or run the pre-extraction command separately for each group.

## Run D2O+ADAPT

Online:

```bash
DATA_ROOT=/path/to/data bash scripts/run_d2o_adapt_online_paper_presets.sh
```

Transductive:

```bash
DATA_ROOT=/path/to/data bash scripts/run_d2o_adapt_transductive_paper_presets.sh
```

The preset scripts apply the paper hyperparameter settings and write logs under `outputs/d2o_adapt_online_paper_presets/` and `outputs/d2o_adapt_transductive_paper_presets/`.

## Notes

- This release does not include result folders, generated class embeddings, cached visual features, or model checkpoints.
- CLIP weights are downloaded by the CLIP loader if they are not already cached.
- Set `CUDA_VISIBLE_DEVICES` outside the scripts if you want to choose a specific GPU.
- The PUG-to-ImageNet class mapping metadata is included at `data/PUG_ImageNet-Class_Ref_ImageNet_class.xlsx`.

## Acknowledgements

This code builds on the public implementations and dataset preparation conventions from:

- [ADAPT](https://github.com/AIM-SKKU/ADAPT)
- [CLIP](https://github.com/openai/CLIP)
- [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp)
- [TPT](https://github.com/azshue/TPT)
- [TDA](https://github.com/kdiAAA/TDA)
- [MTA](https://github.com/MaxZanella/MTA)
- [ZERO](https://github.com/FarinaMatteo/zero)

We thank the authors for releasing their code and dataset preparation instructions.

## Citation

Citation information will be added with the public paper version.
