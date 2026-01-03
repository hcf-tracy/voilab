# Diffusion Policy Training: Single Encoder vs Ensemble Encoder

This repository provides instructions for training diffusion policies for robotic manipulation tasks using two vision encoder configurations in a **Google Colab Environment**:

1. **Single Encoder (FT)**: Standard fine-tuning with a single CLIP-pretrained vision encoder
2. **Ensemble Encoder**: Novel dual-encoder architecture combining ViT-Base and ConvNeXt-Base

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start (Google Colab)](#quick-start-google-colab)
- [Training Configurations](#training-configurations)
  - [Single Encoder (Baseline)](#single-encoder-baseline)
  - [Ensemble Encoder](#ensemble-encoder)
- [Architecture Details](#architecture-details)
- [Citation](#citation)

---

## Overview

We compare two vision encoder architectures for diffusion-based imitation learning:

| Configuration | Vision Encoder | Parameters | Training Time | GPU Memory |
|--------------|----------------|------------|---------------|------------|
| Single (FT) | ViT-Base CLIP | ~160M | ~3 hours | ~10GB |
| Ensemble | ViT-Base + ConvNeXt-Base | ~178M | ~5 hours | ~14GB |

**Key Finding**: The ensemble encoder achieves lower validation error despite higher training loss fluctuation, suggesting better generalization through complementary feature extraction.

---

## Requirements

- Google Colab with GPU runtime (A100)
- Google Drive for dataset storage and checkpoints
- ~15GB free space on Google Drive

### Software Dependencies

The following are automatically installed by the setup script:
- Python 3.10+
- PyTorch 2.0+
- timm (PyTorch Image Models)
- Hydra (configuration management)
- WandB (experiment tracking)

---

## Dataset Preparation

1. Prepare your demonstration dataset in Zarr format
2. Upload `dataset.zarr.zip` to your Google Drive root folder
3. The dataset should contain:
   - RGB observations: `(N, C, H, W)` where `H=W=224`
   - Robot state: end-effector position, rotation, gripper width
   - Actions: target end-effector poses

---

## Quick Start (Google Colab)

### Step 1: Environment Setup

Run this cell to set up the complete environment:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify dataset exists
!ls -lah /content/drive/MyDrive/dataset.zarr.zip

# Install dependencies (takes ~2-3 min)
!apt-get update -y && apt-get install -y cmake build-essential > /dev/null 2>&1
!pip -q install uv

# Clone repository
!git clone https://github.com/author31/voilab.git 2>/dev/null || echo "Already cloned"
%cd /content/voilab
!make install

# Copy dataset to local storage (FASTER I/O!)
!cp /content/drive/MyDrive/dataset.zarr.zip /content/dataset.zarr.zip
!ls -lah /content/dataset.zarr.zip

print("\n✅ Setup complete!")
```

### Step 2: Create Results Directory

```python
import os
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path('/content/drive/MyDrive/diffusion_policy_hpsearch')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Results will be saved to: {RESULTS_DIR}")
```

---

## Training Configurations

### Single Encoder (Baseline)

This configuration uses a single ViT-Base CLIP encoder with aggressive hyperparameters for fast convergence.

**Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 3e-4 | Higher LR for faster convergence with pretrained encoder |
| Batch Size | 128 | Larger batch for stable gradients |
| Epochs | 40 | Early stopping before loss fluctuation |
| EMA Power | 0.8 | Moderate smoothing for policy stability |
| Diffusion Steps | 100 | Standard for action generation |

**Training Command:**

```bash
!uv run packages/diffusion_policy/train.py \
  --config-path=src/diffusion_policy/config \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/content/dataset.zarr.zip \
  training.device=cuda:0 \
  optimizer.lr=3e-4 \
  dataloader.batch_size=128 \
  training.num_epochs=40 \
  ema.power=0.8 \
  policy.noise_scheduler.num_train_timesteps=100 \
  training.val_every=5 \
  training.rollout_every=10 \
  checkpoint.save_last_ckpt=true \
  checkpoint.topk.k=3 \
  hydra.run.dir=/content/drive/MyDrive/diffusion_policy_hpsearch/run_ft
```

**Expected Output:**
- Training time: ~5 hours on A100 GPU
- Final train loss: ~0.10
- Checkpoints saved to: `/content/drive/MyDrive/diffusion_policy_hpsearch/run_ft/checkpoints/`

---

### Ensemble Encoder

This configuration uses our custom `EnsembleObsEncoder` that fuses features from ViT-Base and ConvNeXt-Base.

#### Step 1: Create the Ensemble Encoder Module

Run this cell to create the custom encoder:

```python
encoder_code = '''import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder

class EnsembleObsEncoder(nn.Module):
    """
    Ensemble encoder combining ViT + ConvNeXt features.
    
    Architecture:
        Input Image -> [ViT-Base (768-d)] ----+
                                              |--> Concat (1792-d) -> Linear -> LayerNorm -> GELU -> Output (768-d)
                    -> [ConvNeXt-Base (1024-d)] --+
    
    Key design decisions:
      - Per-timestep fusion: fuse features at each timestep, then pool over time
      - Concatenation fusion: preserves full information from both encoders
      - Learnable projection: allows model to weight encoder contributions
    """

    def __init__(
        self,
        shape_meta: dict,
        # Model 1 settings
        model1_name: str = 'vit_base_patch16_clip_224.openai',
        model1_pretrained: bool = True,
        model1_frozen: bool = False,
        # Model 2 settings
        model2_name: str = 'convnext_base.clip_laion2b_augreg_ft_in12k',
        model2_pretrained: bool = True,
        model2_frozen: bool = False,
        # Fusion settings
        fusion_type: str = 'concat',
        output_dim: int = 768,
        model1_weight: float = 0.5,
        model2_weight: float = 0.5,
        # Shared encoder settings (from original config)
        global_pool: str = '',
        feature_aggregation: str = 'attention_pool_2d',
        position_encording: str = 'sinusoidal',  # Note: typo matches original
        use_group_norm: bool = True,
        share_rgb_model: bool = False,
        imagenet_norm: bool = True,
        downsample_ratio: int = 32,
        transforms: Optional[List] = None,
        # Accept and IGNORE original single-model params to avoid errors
        model_name: str = None,
        pretrained: bool = None,
        frozen: bool = None,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[EnsembleObsEncoder] Ignoring extra kwargs: {list(kwargs.keys())}")

        self.fusion_type = fusion_type
        self.output_dim = output_dim
        self.model1_weight = model1_weight
        self.model2_weight = model2_weight
        self.shape_meta = shape_meta

        # Common encoder kwargs
        enc_kwargs = dict(
            shape_meta=shape_meta,
            global_pool=global_pool,
            feature_aggregation=feature_aggregation,
            position_encording=position_encording,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
            downsample_ratio=downsample_ratio,
            transforms=transforms or [],
        )

        print(f"[EnsembleObsEncoder] Creating encoder 1: {model1_name}")
        self.encoder1 = TimmObsEncoder(
            model_name=model1_name,
            pretrained=model1_pretrained,
            frozen=model1_frozen,
            **enc_kwargs
        )

        print(f"[EnsembleObsEncoder] Creating encoder 2: {model2_name}")
        self.encoder2 = TimmObsEncoder(
            model_name=model2_name,
            pretrained=model2_pretrained,
            frozen=model2_frozen,
            **enc_kwargs
        )

        # Get ACTUAL per-step feature dimensions via dummy forward pass
        self.feat_dim1, self.feat_dim2 = self._get_feature_dims()
        print(f"[EnsembleObsEncoder] Feature dims (per-step): model1={self.feat_dim1}, model2={self.feat_dim2}")

        # Build fusion layers (operate on last-dim only)
        if fusion_type == 'concat':
            total_dim = self.feat_dim1 + self.feat_dim2
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            self._output_dim = output_dim
            print(f"[EnsembleObsEncoder] Concat fusion: {self.feat_dim1}+{self.feat_dim2}={total_dim} -> {output_dim}")

        elif fusion_type == 'average':
            self.proj1 = nn.Linear(self.feat_dim1, output_dim)
            self.proj2 = nn.Linear(self.feat_dim2, output_dim)
            self.norm = nn.LayerNorm(output_dim)
            self._output_dim = output_dim
            print(f"[EnsembleObsEncoder] Average fusion: {self.feat_dim1}, {self.feat_dim2} -> {output_dim}")

        elif fusion_type == 'attention':
            self.proj1 = nn.Linear(self.feat_dim1, output_dim)
            self.proj2 = nn.Linear(self.feat_dim2, output_dim)
            self.cross_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(output_dim)
            self._output_dim = output_dim
            print(f"[EnsembleObsEncoder] Attention fusion -> {output_dim}")

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EnsembleObsEncoder] Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    def _make_dummy_obs(self, T: int = 2) -> Dict[str, torch.Tensor]:
        """
        Build dummy input matching shape_meta.
        Use T=2 by default to mimic training (your error shows T=2).
        """
        dummy_obs = {}
        obs_shape_meta = self.shape_meta.get('obs', {})
        for key, attr in obs_shape_meta.items():
            shape = attr.get('shape', [])
            obs_type = attr.get('type', 'low_dim')

            if obs_type == 'rgb':
                c, h, w = shape
                dummy_obs[key] = torch.zeros(1, T, c, h, w)
            elif obs_type == 'low_dim':
                d = shape[0] if shape else 1
                dummy_obs[key] = torch.zeros(1, T, d)
        return dummy_obs

    def _get_feature_dims(self) -> Tuple[int, int]:
        """
        Get per-step output dims via dummy forward pass.
        If encoder returns (B,T,D), return D (NOT T*D).
        """
        dummy_obs = self._make_dummy_obs(T=2)
        with torch.no_grad():
            feat1 = self.encoder1(dummy_obs)
            feat2 = self.encoder2(dummy_obs)

        # Normalize to (B,T,D) or (B,D)
        if feat1.dim() == 3:
            dim1 = feat1.shape[-1]
        elif feat1.dim() == 2:
            dim1 = feat1.shape[-1]
        else:
            dim1 = feat1.numel()

        if feat2.dim() == 3:
            dim2 = feat2.shape[-1]
        elif feat2.dim() == 2:
            dim2 = feat2.shape[-1]
        else:
            dim2 = feat2.numel()

        return dim1, dim2

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass:
          - encoder outputs may be (B,D) or (B,T,D)
          - fuse per-timestep if T exists
          - pool over time -> (B, output_dim)
        """
        feat1 = self.encoder1(obs_dict)
        feat2 = self.encoder2(obs_dict)

        # Case A: both are (B,D)
        if feat1.dim() == 2 and feat2.dim() == 2:
            if self.fusion_type == 'concat':
                combined = torch.cat([feat1, feat2], dim=-1)           # (B, D1+D2)
                out = self.fusion(combined)                           # (B, out)
            elif self.fusion_type == 'average':
                p1 = self.proj1(feat1)
                p2 = self.proj2(feat2)
                out = self.model1_weight * p1 + self.model2_weight * p2
                out = self.norm(out)
            else:  # attention
                q = self.proj1(feat1).unsqueeze(1)                    # (B,1,out)
                kv = self.proj2(feat2).unsqueeze(1)                   # (B,1,out)
                attn_out, _ = self.cross_attn(q, kv, kv)
                out = self.norm(q + attn_out).squeeze(1)              # (B,out)
            return out

        # Case B: at least one is (B,T,D) -> convert both to (B,T,D)
        if feat1.dim() == 2:
            feat1 = feat1.unsqueeze(1)                                # (B,1,D1)
        if feat2.dim() == 2:
            feat2 = feat2.unsqueeze(1)                                # (B,1,D2)

        # Align T by broadcasting if needed
        T1 = feat1.shape[1]
        T2 = feat2.shape[1]
        if T1 != T2:
            if T1 == 1 and T2 > 1:
                feat1 = feat1.expand(-1, T2, -1)
            elif T2 == 1 and T1 > 1:
                feat2 = feat2.expand(-1, T1, -1)
            else:
                raise RuntimeError(f"[EnsembleObsEncoder] Time dim mismatch: T1={T1}, T2={T2}")

        # Now both (B,T,D)
        if self.fusion_type == 'concat':
            combined = torch.cat([feat1, feat2], dim=-1)              # (B,T,D1+D2)
            out = self.fusion(combined)                               # (B,T,out)
            out = out.mean(dim=1)                                     # (B,out)

        elif self.fusion_type == 'average':
            p1 = self.proj1(feat1)                                    # (B,T,out)
            p2 = self.proj2(feat2)                                    # (B,T,out)
            out = self.model1_weight * p1 + self.model2_weight * p2
            out = self.norm(out)
            out = out.mean(dim=1)                                     # (B,out)

        else:  # attention
            # Use feat1 as query, feat2 as key/value per timestep
            q = self.proj1(feat1)                                     # (B,T,out)
            kv = self.proj2(feat2)                                    # (B,T,out)
            attn_out, _ = self.cross_attn(q, kv, kv)                  # (B,T,out)
            out = self.norm(q + attn_out).mean(dim=1)                 # (B,out)

        return out

    def output_shape(self) -> Tuple[int]:
        return (self._output_dim,)
'''

file_path = '/content/voilab/packages/diffusion_policy/src/diffusion_policy/model/vision/ensemble_obs_encoder.py'
with open(file_path, 'w') as f:
    f.write(encoder_code)

print("✅ Fixed ensemble encoder (time-safe fusion) created!")
```

#### Step 2: Run Ensemble Training

**Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 1e-4 | Lower LR for dual-encoder stability |
| Batch Size | 32 | Reduced due to GPU memory constraints |
| Gradient Accumulation | 2 | Effective batch size = 64 |
| Epochs | 60 | More epochs for ensemble convergence |
| EMA Power | 0.75 | Stronger smoothing for complex model |
| LR Warmup | 1000 steps | Gradual warmup for fusion layer adaptation |

**Data Augmentation:**
| Augmentation | Value | Purpose |
|--------------|-------|---------|
| Random Crop Ratio | 0.85 | Position variance |
| Brightness | ±0.4 | Lighting robustness |
| Contrast | ±0.4 | Lighting robustness |
| Saturation | ±0.5 | Color variation |
| Hue | ±0.1 | Color variation |
| Input Perturbation | 0.1 | Diffusion noise robustness |

**Training Command:**

```bash
!uv run packages/diffusion_policy/train.py \
  --config-path=src/diffusion_policy/config \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/content/dataset.zarr.zip \
  training.device=cuda:0 \
  "policy.obs_encoder._target_=diffusion_policy.model.vision.ensemble_obs_encoder.EnsembleObsEncoder" \
  "+policy.obs_encoder.model1_name=vit_base_patch16_clip_224.openai" \
  +policy.obs_encoder.model1_pretrained=True \
  +policy.obs_encoder.model1_frozen=False \
  "+policy.obs_encoder.model2_name=convnext_base.clip_laion2b_augreg_ft_in12k" \
  +policy.obs_encoder.model2_pretrained=True \
  +policy.obs_encoder.model2_frozen=False \
  "+policy.obs_encoder.fusion_type=concat" \
  +policy.obs_encoder.output_dim=768 \
  policy.obs_encoder.transforms.0.ratio=0.85 \
  policy.obs_encoder.transforms.1.brightness=0.4 \
  policy.obs_encoder.transforms.1.contrast=0.4 \
  policy.obs_encoder.transforms.1.saturation=0.5 \
  policy.obs_encoder.transforms.1.hue=0.1 \
  policy.input_pertub=0.1 \
  policy.noise_scheduler.num_train_timesteps=100 \
  optimizer.lr=1e-4 \
  optimizer.weight_decay=1e-5 \
  dataloader.batch_size=32 \
  training.gradient_accumulate_every=2 \
  training.num_epochs=45 \
  training.lr_warmup_steps=1000 \
  training.val_every=5 \
  training.rollout_every=10 \
  ema.power=0.75 \
  checkpoint.save_last_ckpt=true \
  checkpoint.topk.k=5 \
  checkpoint.topk.monitor_key=train_loss \
  checkpoint.topk.mode=min \
  hydra.run.dir=/content/drive/MyDrive/diffusion_policy_hpsearch/fork_knife_ensemble_v2
```

**Expected Output:**
- Training time: ~10 hours on A100 GPU
- Final train loss: 0.02-0.05 (with fluctuation)
- Checkpoints saved to: `/content/drive/MyDrive/diffusion_policy_hpsearch/fork_knife_ensemble_v2/checkpoints/`

---

## Architecture Details

### Single Encoder Architecture

```
Input (B, T, 3, 224, 224)
         │
         ▼
┌─────────────────────┐
│   ViT-Base CLIP     │
│   86M parameters    │
│   768-dim output    │
└─────────────────────┘
         │
         ▼
    Features (B, 768)
         │
         ▼
┌─────────────────────┐
│  Diffusion U-Net    │
│   74M parameters    │
└─────────────────────┘
         │
         ▼
   Actions (B, H, A)
```

### Ensemble Encoder Architecture

```
Input (B, T, 3, 224, 224)
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ViT-Base│ │ConvNeXt-Base│
│  CLIP  │ │    CLIP     │
│ 86M    │ │    92M      │
│ 768-d  │ │   1024-d    │
└────────┘ └────────────┘
    │             │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Concatenate │
    │   1792-dim   │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   Linear    │
    │ 1792 → 768  │
    │  LayerNorm  │
    │    GELU     │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Mean Pool   │
    │   (time)    │
    └─────────────┘
           │
           ▼
    Features (B, 768)
           │
           ▼
    ┌─────────────┐
    │Diffusion    │
    │  U-Net      │
    └─────────────┘
           │
           ▼
    Actions (B, H, A)
```

### Why Two Encoders?

| Encoder | Strength | Mechanism |
|---------|----------|-----------|
| **ViT-Base** | Global context understanding | Self-attention captures long-range dependencies |
| **ConvNeXt-Base** | Local feature extraction | Convolutional inductive bias for spatial details |

The combination provides:
- **Semantic understanding** (ViT): Recognizes object types and scene context
- **Spatial precision** (ConvNeXt): Accurate localization for manipulation

---

## Results

### Training Curves

| Metric | Single Encoder | Ensemble |
|--------|---------------|----------|
| Training Loss | Smooth convergence | Fluctuating |
| Validation Error | Higher | Lower |
| Convergence Speed | Faster | Slower |

### Key Observations

1. **Training Stability**: Single encoder shows smoother loss curves
2. **Generalization**: Ensemble achieves lower validation error despite noisy training
3. **Implicit Regularization**: Training fluctuation may prevent overfitting

---

## File Structure

After training, your Google Drive should contain:

```
/content/drive/MyDrive/diffusion_policy_hpsearch/
├── run_ft/                          # Single encoder results
│   ├── checkpoints/
│   │   ├── latest.ckpt
│   │   └── top_k/
│   ├── config.yaml
│   └── train.log
│
└── fork_knife_ensemble_v2/          # Ensemble results
    ├── checkpoints/
    │   ├── latest.ckpt
    │   └── top_k/
    ├── config.yaml
    └── train.log
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{chi2024diffusionpolicyvisuomotorpolicy,
      title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion}, 
      author={Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
      year={2024},
      eprint={2303.04137},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2303.04137}, 
}
```
- author31. *voilab*. GitHub repository.  
  https://github.com/author31/voilab


---
