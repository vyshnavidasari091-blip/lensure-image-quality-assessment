"""
Synthetic training-data generation for the deep-learning head.

Methodology (standard in the NR-IQA literature, e.g. LIVE / TID2013 / KADID-10k):
  1. Start from "pristine" (clean) natural-looking images.
  2. Apply controlled, known degradations at several severities.
  3. Train a model to predict the degradation type/severity from the
     degraded image only (no reference) -> this transfers to unseen real
     photos at inference time.

To keep the whole pipeline runnable with zero external downloads (no
internet dataset dependency, no external AI APIs), pristine images are
generated procedurally: photographic-like scenes are simulated using
smooth gradients, Perlin-ish noise textures, and randomly placed shapes,
which give the natural-image statistics (smooth low-frequency regions +
textured/edge regions) that the feature extractors and CNN key on.

For a production submission, swap `make_pristine_image` for images drawn
from a real dataset (e.g. a folder of the applicant's own clean photos,
or a public dataset such as DIV2K / KADID-10k pristine set) -- the
degradation and training code is agnostic to where the clean images
come from.
"""
from __future__ import annotations
import numpy as np
import cv2

RNG = np.random.default_rng(42)

LABELS = ["clean", "blur", "underexposed", "overexposed", "noise", "corrupt"]


def make_pristine_image(size: int = 224, rng: np.random.Generator = RNG) -> np.ndarray:
    """Procedurally synthesize a natural-looking, *sharp* BGR image (uint8).

    Mixes a smooth low-frequency base (sky/wall-like gradients) with
    crisp high-frequency content (fine texture + hard-edged shapes +
    sensor-like micro-texture) so the pristine class sits at a realistic,
    clearly-in-focus sharpness level -- this matters because the `blur`
    degradation class, and the blur detector's threshold, are calibrated
    relative to this baseline.
    """
    # low-frequency smooth base (simulates sky/wall/skin gradients)
    small = rng.normal(loc=128, scale=40, size=(size // 16, size // 16, 3)).astype(np.float32)
    base = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)

    img = base.copy()

    # crisp hard-edged shapes (objects, no anti-aliasing) -> real high-frequency edges
    n_shapes = rng.integers(6, 14)
    for _ in range(n_shapes):
        color = tuple(int(c) for c in rng.integers(10, 245, size=3))
        x, y = rng.integers(0, size, size=2)
        r = rng.integers(6, size // 4)
        if rng.random() < 0.5:
            cv2.circle(img, (int(x), int(y)), int(r), color, -1, lineType=cv2.LINE_4)
        else:
            x2, y2 = x + rng.integers(-r, r), y + rng.integers(-r, r)
            cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), color, -1)

    # fine speckle texture (simulates foliage/fabric/sensor grain, high-frequency)
    fine_noise = rng.normal(loc=0, scale=14, size=(size, size, 3)).astype(np.float32)
    img = img + fine_noise

    # a sprinkling of thin high-contrast lines for extra crisp edge content
    n_lines = rng.integers(4, 10)
    for _ in range(n_lines):
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.line(img, p1, p2, color, thickness=1, lineType=cv2.LINE_4)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def degrade(img: np.ndarray, label: str, severity: float, rng: np.random.Generator = RNG) -> np.ndarray:
    """severity in [0, 1]; 0 = imperceptible, 1 = extreme."""
    out = img.copy()
    if label == "clean":
        return out
    if label == "blur":
        k = max(1, int(2 + severity * 14))
        k = k + 1 if k % 2 == 0 else k
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif label == "underexposed":
        factor = 1.0 - 0.85 * severity
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif label == "overexposed":
        factor = 1.0 + 2.5 * severity
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif label == "noise":
        sigma = 5 + severity * 60
        noise = rng.normal(0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif label == "corrupt":
        # simulate block corruption / severe compression + row dropout
        h, w = out.shape[:2]
        n_blocks = int(2 + severity * 10)
        for _ in range(n_blocks):
            bh, bw = rng.integers(5, max(6, h // 6)), rng.integers(5, max(6, w // 6))
            y, x = rng.integers(0, max(1, h - bh)), rng.integers(0, max(1, w - bw))
            out[y:y + bh, x:x + bw] = rng.integers(0, 255, size=(bh, bw, 3))
        q = int(max(2, 40 - severity * 35))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def synth_batch(n: int, size: int = 224, rng: np.random.Generator = RNG):
    """Yield (image, label, severity) for training."""
    for _ in range(n):
        base = make_pristine_image(size, rng)
        label = rng.choice(LABELS)
        severity = 0.0 if label == "clean" else float(rng.uniform(0.15, 1.0))
        img = degrade(base, label, severity, rng)
        yield img, label, severity
