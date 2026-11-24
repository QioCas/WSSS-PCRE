"""Conservative pseudo-label expansion that never rescales foreground CAMs.

Preserve baseline PAR foreground/background assignments. Admit a previously
ignored pixel only when the main CAM, class prototype, and PAR's low-background
branch agree on its class. No pixel ground truth is accepted by this module.
"""
import torch

from utils.adaptive_threshold import AdaptiveCAMThreshold


class CenteredCAMThreshold(AdaptiveCAMThreshold):
    """Class-balanced relative confidence centered on the original threshold.

Unlike the legacy global-confidence product, uniformly lower CAM scores do not
lower every class threshold. Bounds and scheduling limit any relaxation.
"""
    @torch.no_grad()
    def thresholds(self, iteration):
        seen = self.observations > 0
        if not seen.any() or iteration < self.warmup:
            return torch.full_like(self.class_confidence, self.base)
        normalizer = self.class_confidence[seen].mean().clamp_min(1e-6)
        target = (self.base*self.class_confidence/normalizer).clamp(self.minimum,self.maximum)
        target = torch.where(seen, target, self.base)
        progress = 1.0 if self.ramp == 0 else min(1.0, (iteration-self.warmup)/self.ramp)
        return self.base+progress*(target-self.base)


@torch.no_grad()
def expand_consensus_labels(baseline, low_labels, cams, prototype_cams,
                            image_labels, thresholds, base=0.7,
                            prototype_threshold=0.7, ignore_index=255):
    """Return a detached label tensor and an explicit added-pixel mask.

    Outside valid crops, low_labels stays ignore, preventing padded pixels
    from passing the admission gate. Foreground class competition is unchanged.
    """
    if cams.shape != prototype_cams.shape:
        raise ValueError('Main and prototype CAM shapes must match')
    if thresholds.shape != (cams.shape[1],):
        raise ValueError('Expected one threshold per foreground class')
    present=image_labels.to(cams.device).bool()[:,:,None,None]
    main=cams.detach().masked_fill(~present,-1)
    proto=prototype_cams.detach().masked_fill(~present,-1)
    main_score,main_class=main.max(1)
    proto_score,proto_class=proto.max(1)
    candidate=main_class+1
    tau=thresholds[main_class]
    added=((baseline==ignore_index) & (low_labels==candidate)
           & (main_class==proto_class) & (proto_score>=prototype_threshold)
           & (main_score>=tau) & (tau<base-1e-6))
    output=baseline.detach().clone()
    output[added]=candidate[added].to(output.dtype)
    return output,added
