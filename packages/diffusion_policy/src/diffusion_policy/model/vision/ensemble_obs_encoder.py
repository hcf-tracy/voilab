# voilab/packages/diffusion_policy/src/diffusion_policy/model/vision/ensemble_obs_encoder.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder

class EnsembleObsEncoder(nn.Module):
    """
    Ensemble encoder combining ViT + ConvNeXt features.

    Key fix:
      - Do NOT flatten time dimension into feature dimension.
      - Fuse per-timestep, then pool over time -> stable input dim for fusion.
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
