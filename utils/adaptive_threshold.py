"""CAM-score adaptation inspired by FreeMatch SAT, not the full SSL algorithm.

Only the main segmentation pseudo-label branch is changed. CAM scores are not
calibrated class probabilities. Statistics exclude padding, absent image classes
and low-activation pixels; no segmentation ground truth is consumed here.
"""
import torch
from torch import nn
import torch.distributed as dist


class AdaptiveCAMThreshold(nn.Module):
    def __init__(self, num_classes, base=0.7, low=0.25, momentum=0.99,
                 minimum=0.55, maximum=0.85, warmup=2000, ramp=1000):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not 0 < low < minimum <= base <= maximum <= 1:
            raise ValueError("require 0 < low < min <= base <= max <= 1")
        if warmup < 0 or ramp < 0:
            raise ValueError("warmup/ramp cannot be negative")
        self.base, self.low, self.momentum = base, low, momentum
        self.minimum, self.maximum = minimum, maximum
        self.warmup, self.ramp = warmup, ramp
        self.register_buffer("global_confidence", torch.tensor(float(base)))
        self.register_buffer("class_confidence", torch.full((num_classes,), float(base)))
        self.register_buffer("observations", torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, cams, image_labels, boxes):
        scores = cams.detach().float().clamp(0, 1)
        present = image_labels.to(scores.device).bool()
        scores = scores.masked_fill(~present[:, :, None, None], -1)
        confidence, winner = scores.max(dim=1)
        valid = torch.zeros_like(confidence, dtype=torch.bool)
        for i, box in enumerate(boxes):
            top, bottom, left, right = [int(v) for v in box]
            valid[i, top:bottom, left:right] = True
        valid &= confidence > self.low
        selected, weights = winner[valid], confidence[valid]
        counts = torch.bincount(selected, minlength=self.class_confidence.numel()).float()
        sums = torch.zeros_like(counts).scatter_add_(0, selected, weights)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        active = counts > 0
        m = self.momentum
        if active.any():
            self.global_confidence.mul_(m).add_((1 - m) * sums.sum() / counts.sum())
            self.class_confidence[active] = (
                m * self.class_confidence[active] + (1 - m) * sums[active] / counts[active]
            )
            self.observations.add_(counts)

    @torch.no_grad()
    def thresholds(self, iteration):
        seen = self.observations > 0
        if not seen.any() or iteration < self.warmup:
            return torch.full_like(self.class_confidence, self.base)
        normalizer = self.class_confidence[seen].max().clamp_min(1e-6)
        target = (self.global_confidence * self.class_confidence / normalizer).clamp(
            self.minimum, self.maximum)
        target = torch.where(seen, target, self.base)
        progress = 1.0 if self.ramp == 0 else min(1.0, (iteration - self.warmup) / self.ramp)
        return self.base + progress * (target - self.base)

    @torch.no_grad()
    def adapt(self, cams, image_labels, boxes, iteration):
        self.update(cams, image_labels, boxes)
        thresholds = self.thresholds(iteration)
        # Scaling before PAR preserves the original refinement implementation.
        # Before refinement it corresponds to high=tau_c, low=low*tau_c/base.
        return cams * (self.base / thresholds)[None, :, None, None]
