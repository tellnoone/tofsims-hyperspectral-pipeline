"""
normalize.py
Normalization routines and signal conditioning for ToF-SIMS stacks.
"""
import numpy as np

def normalize_tic(stack: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    tic = np.sum(stack, axis=-1, keepdims=True)
    tic[tic == 0] = eps
    return stack / tic

def min_max_scale(stack: np.ndarray) -> np.ndarray:
    min_val = np.min(stack, axis=(0, 1), keepdims=True)
    max_val = np.max(stack, axis=(0, 1), keepdims=True)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (stack - min_val) / range_val

def clip_outliers(stack: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    clipped_stack = stack.copy()
    for ch in range(stack.shape[-1]):
        cutoff = np.percentile(stack[:, :, ch], percentile)
        clipped_stack[:, :, ch] = np.clip(stack[:, :, ch], 0, cutoff)
    return clipped_stack