# Dream-IF: Dynamic Relative EnhAnceMent for Image Fusion

[![arXiv](https://img.shields.io/badge/arXiv-2503.10109-b31b1b.svg)](https://arxiv.org/abs/2503.10109)

> **Dream-IF: Dynamic Relative EnhAnceMent for Image Fusion**  
> Xingxin Xu, Bing Cao, Dongdong Li, Qinghua Hu, Pengfei Zhu  
> *arXiv 2025*

## Introduction

Image fusion aims to integrate comprehensive information from images acquired through multiple sources. However, images captured by diverse sensors often encounter various degradations that can negatively affect fusion quality. Traditional fusion methods generally treat image enhancement and fusion as separate processes, overlooking the inherent correlation between them — notably, the dominant regions in one modality of a fused image often indicate areas where the other modality might benefit from enhancement.

Inspired by this observation, we introduce the concept of **dominant regions** for image enhancement and present **Dream-IF**, a Dynamic Relative EnhAnceMent framework for Image Fusion. This framework:

- Quantifies the **relative dominance** of each modality across different layers and leverages this information to facilitate reciprocal cross-modal enhancement.
- Supports not only image restoration but also a broader range of **image enhancement** applications by integrating the relative dominance derived from image fusion.
- Employs **prompt-based encoding** to capture degradation-specific details, which dynamically steer the restoration process and promote coordinated enhancement in both multi-modal image fusion and image enhancement scenarios.

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

## Repository Structure

```
Dream-IF/
├── train.py                # Training script
├── test.py                 # Testing / inference script
├── option.py               # Training argument parser
├── losses.py               # Fusion loss functions
├── transforms.py           # Data augmentation transforms
├── networks/
│   ├── restormer.py        # Main backbone (Restormer-based)
│   ├── models.py           # Model definitions
│   ├── MMOE.py             # Multi-gate Mixture-of-Experts module
│   ├── TC_MoA.py           # Task-Customized MoA module
│   ├── ViT_MAE.py          # Vision Transformer / MAE components
│   ├── vit_model.py        # ViT model utilities
│   ├── Encoder.py          # Encoder module
│   └── Windows_Shift.py    # Window-based shift attention
├── data/
│   ├── dataloader_VIF.py   # Dataloader for visible-infrared fusion
│   ├── dateloader.py       # General custom dataset loader
│   └── dateloader_test.py  # Test dataset loader
├── util/                   # MAE / training utilities
│   ├── fusion_loss.py      # Fusion loss components (SSIM, gradient, pixel)
│   ├── mefssim.py          # MEF-MSSSIM metric
│   ├── misc.py             # Distributed training helpers
│   ├── pos_embed.py        # Positional embedding utilities
│   ├── transforms.py       # Image transform pipelines
│   └── ...
└── utils/                  # Image processing utilities
    ├── degeneration.py     # Degradation simulation
    ├── fusion_loss.py      # Loss components
    ├── utils.py            # General utilities
    ├── utils_image.py      # Image I/O and manipulation
    └── utils_blindsr.py    # Blind super-resolution utilities
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA (recommended)

```bash
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `torchvision`, `timm`, `einops`, `opencv-python`, `numpy`, `Pillow`

## Dataset Preparation

Prepare your dataset with the following structure:

```
dataset/
├── vi/          # Visible images
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── ir/          # Infrared images
    ├── 001.png
    ├── 002.png
    └── ...
```

## Training

```bash
python train.py \
    --data_root /path/to/training/data \
    --save_name experiment_name \
    --batch_size 4 \
    --lr 0.0003 \
    --niter 200 \
    --gpu_id 0
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `''` | Path to training data |
| `--batch_size` | `4` | Training batch size |
| `--lr` | `0.0003` | Learning rate |
| `--niter` | `200` | Number of training epochs |
| `--image_size` | `286` | Input image size |
| `--gpu_id` | `'0'` | GPU device ID |
| `--save_name` | `'first'` | Experiment name |
| `--resume` | `False` | Resume training from checkpoint |
| `--resume_pth` | `''` | Path to checkpoint for resuming |

## Testing

```bash
python test.py \
    --dataset_path /path/to/test/data \
    --weights_path /path/to/checkpoint.pth \
    --save_path ./results \
    --gpu_id 0
```

The test dataset should follow the same `vi/` and `ir/` directory structure. Results will be saved to `--save_path`.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{xu2025dreamif,
  title     = {Dream-IF: Dynamic Relative EnhAnceMent for Image Fusion},
  author    = {Xu, Xingxin and Cao, Bing and Li, Dongdong and Hu, Qinghua and Zhu, Pengfei},
  journal   = {arXiv preprint arXiv:2503.10109},
  year      = {2025}
}
```

## Acknowledgements

This codebase builds upon:
- [Restormer](https://github.com/swz30/Restormer) — Efficient Transformer for High-Resolution Image Restoration
- [MAE](https://github.com/facebookresearch/mae) — Masked Autoencoders
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models

## License

This project is released for academic research use. Please refer to the licenses of the respective components.

## Contact

If you have any questions, please open an issue or contact the authors.
