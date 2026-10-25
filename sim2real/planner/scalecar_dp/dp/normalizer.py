"""Mean/std normalization of motion chunks (per chunk position and dim) and history (per sample and dim),
estimated on training samples only and stored with every checkpoint."""
from __future__ import annotations

import numpy as np
import torch

MIN_STD = 1e-3


class Normalizer(torch.nn.Module):
    def __init__(self, target_mean, target_std, hist_mean, hist_std):
        super().__init__()
        for k, v in dict(target_mean=target_mean, target_std=target_std, hist_mean=hist_mean, hist_std=hist_std).items():
            self.register_buffer(k, torch.as_tensor(np.asarray(v), dtype=torch.float32))

    @classmethod
    def fit(cls, targets: np.ndarray, histories: np.ndarray, hist_masks: np.ndarray) -> 'Normalizer':
        """targets (N,T,38), histories (N,H,41), hist_masks (N,H) bool."""
        tm, ts = targets.mean(0), np.maximum(targets.std(0), MIN_STD)
        w = hist_masks[..., None].astype(np.float64)
        hm = (histories * w).sum(0) / np.maximum(w.sum(0), 1)
        hs = np.sqrt(((histories - hm) ** 2 * w).sum(0) / np.maximum(w.sum(0), 1))
        return cls(tm, ts, hm, np.maximum(hs, MIN_STD))

    def norm_target(self, x):
        return (x - self.target_mean) / self.target_std

    def denorm_target(self, x):
        return x * self.target_std + self.target_mean

    def norm_history(self, h, mask):
        return (h - self.hist_mean) / self.hist_std * mask.unsqueeze(-1)
