"""
Classical, interpretable image-quality features.

These are the same family of statistics used across the no-reference
image-quality-assessment (NR-IQA) literature (e.g. spatial-domain natural
scene statistics as in BRISQUE, the Laplacian-variance sharpness measure,
and the fast single-image noise estimator of Immerkaer 1996). They are
cheap to compute, fully explainable, and used both as standalone signals
and as engineered input features to the learned model.
"""
from __future__ import annotations
import numpy as np
import cv2


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Low values => blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(gray: np.ndarray) -> dict:
    mean = float(gray.mean())
    std = float(gray.std())
    over_frac = float(np.mean(gray >= 250))
    under_frac = float(np.mean(gray <= 5))
    return {
        "brightness_mean": mean,
        "contrast_std": std,
        "overexposed_fraction": over_frac,
        "underexposed_fraction": under_frac,
    }


def noise_sigma(gray: np.ndarray) -> float:
    """
    Fast single-image noise-standard-deviation estimator (Immerkaer, 1996):
    convolve with a Laplacian-of-Gaussian-like mask that cancels flat and
    edge structure, leaving a signal dominated by noise, then rescale.
    """
    H, W = gray.shape
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, mask)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2) + 1e-6)
    return float(sigma)


def colorfulness(img_bgr: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


def corruption_signals(img_bgr: np.ndarray) -> dict:
    """Cheap heuristics for truncated/garbled/degenerate images."""
    gray = _to_gray(img_bgr)
    h, w = gray.shape
    flat_fraction = float(np.mean(np.abs(np.diff(gray.astype(np.int16), axis=1)) == 0))
    degenerate_size = bool(h < 32 or w < 32)
    return {"flat_row_fraction": flat_fraction, "degenerate_size": degenerate_size}


def extract_all(img_bgr: np.ndarray) -> dict:
    gray = _to_gray(img_bgr)
    feats = {
        "sharpness": sharpness_score(gray),
        "noise_sigma": noise_sigma(gray),
        "colorfulness": colorfulness(img_bgr),
        "entropy": entropy(gray),
        "edge_density": edge_density(gray),
    }
    feats.update(brightness_stats(gray))
    feats.update(corruption_signals(img_bgr))
    return feats


def feature_vector(img_bgr: np.ndarray) -> np.ndarray:
    """Fixed-order numeric vector, used as the classical-feature input to the ML head."""
    f = extract_all(img_bgr)
    order = [
        "sharpness", "noise_sigma", "colorfulness", "entropy", "edge_density",
        "brightness_mean", "contrast_std", "overexposed_fraction",
        "underexposed_fraction", "flat_row_fraction",
    ]
    return np.array([f[k] for k in order], dtype=np.float32)
