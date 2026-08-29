"""
Hybrid image-quality model.

Design choice (see README section "Model selection"): a *lightweight CNN
trained from scratch*, fused with the engineered classical features from
features.py, rather than an ImageNet transfer-learning backbone. This is
one of the four approaches explicitly allowed by the assessment ("a
lightweight deep-learning model ... or a hybrid approach combining
image-quality features with a learned model") and was chosen because:
  - it needs no external pretrained-weight download (fully offline/
    reproducible, no external service dependency),
  - it trains in a couple of minutes on CPU, which matters for a 48h
    assessment window,
  - the classical features already carry strong, interpretable signal
    for this task, so a small CNN only needs to learn complementary
    texture/context cues, not relearn ImageNet-level representations.

Two heads:
  - `class_logits` (6-way): clean / blur / underexposed / overexposed / noise / corrupt
  - `quality`      (1, sigmoid*100): overall quality score
"""
from __future__ import annotations
import torch
import torch.nn as nn

from app.ml.synth_data import LABELS

N_CLASSES = len(LABELS)
N_CLASSICAL_FEATS = 10
IMG_SIZE = 224


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridIQAModel(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, n_classical: int = N_CLASSICAL_FEATS):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_norm = nn.BatchNorm1d(n_classical)
        fusion_dim = 128 + n_classical
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.class_head = nn.Linear(96, n_classes)
        self.quality_head = nn.Linear(96, 1)

    def forward(self, img: torch.Tensor, classical_feats: torch.Tensor):
        x = self.backbone(img).flatten(1)                 # (B, 128)
        cf = self.feat_norm(classical_feats)               # (B, n_classical)
        fused = self.fusion(torch.cat([x, cf], dim=1))
        logits = self.class_head(fused)
        quality = torch.sigmoid(self.quality_head(fused)).squeeze(1) * 100.0
        return logits, quality
