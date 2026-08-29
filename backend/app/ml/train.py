"""
Train the hybrid IQA model on procedurally-generated synthetic distortions
and evaluate on a held-out synthetic split. Run:

    python -m app.ml.train

Produces:
  app/ml/weights/iqa_head.pt   (model weights)
  app/ml/weights/metrics.json  (evaluation metrics + confusion matrix)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from app.ml.synth_data import synth_batch, LABELS
from app.ml.features import feature_vector
from app.ml.model import HybridIQAModel, IMG_SIZE

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


class SynthDataset(Dataset):
    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.samples = list(synth_batch(n, size=IMG_SIZE, rng=rng))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bgr, label, severity = self.samples[idx]
        feats = feature_vector(img_bgr)
        img_t = torch.from_numpy(img_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        cls_idx = LABEL_TO_IDX[label]
        # ground-truth quality: 100 for clean, degrading with severity otherwise
        quality = 100.0 if label == "clean" else max(0.0, 100.0 * (1.0 - severity) * 0.9)
        return img_t, torch.from_numpy(feats), cls_idx, quality


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    feats = torch.stack([b[1] for b in batch])
    cls = torch.tensor([b[2] for b in batch], dtype=torch.long)
    qual = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return imgs, feats, cls, qual


def train(n_train=1200, n_val=300, epochs=6, batch_size=32, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    train_ds = SynthDataset(n_train, seed=seed)
    val_ds = SynthDataset(n_val, seed=seed + 999)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = HybridIQAModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, feats, cls, qual in train_dl:
            opt.zero_grad()
            logits, quality = model(imgs, feats)
            loss = ce(logits, cls) + 0.01 * mse(quality, qual)
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
        print(f"epoch {epoch+1}/{epochs}  train_loss={running/len(train_ds):.4f}")

    print(f"training took {time.time()-t0:.1f}s")

    # ---- evaluation on held-out synthetic split ----
    model.eval()
    all_preds, all_true, all_qpred, all_qtrue = [], [], [], []
    with torch.no_grad():
        for imgs, feats, cls, qual in val_dl:
            logits, quality = model(imgs, feats)
            preds = logits.argmax(1)
            all_preds += preds.tolist()
            all_true += cls.tolist()
            all_qpred += quality.tolist()
            all_qtrue += qual.tolist()

    acc = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(LABELS)))).tolist()
    mae_quality = float(np.mean(np.abs(np.array(all_qpred) - np.array(all_qtrue))))

    metrics = {
        "n_train": n_train,
        "n_val": n_val,
        "epochs": epochs,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "quality_mae": mae_quality,
        "labels": LABELS,
        "confusion_matrix": cm,
    }
    print(json.dumps(metrics, indent=2))

    torch.save(model.state_dict(), WEIGHTS_DIR / "iqa_head.pt")
    with open(WEIGHTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved weights + metrics to {WEIGHTS_DIR}")


if __name__ == "__main__":
    train()
