import torch
from torch import nn
from torch.nn import functional as F

def _boundary_weight_map(
    targets: torch.Tensor,
    boundary_weight: float,
    boundary_radius: int = 3,
    boundary_class: int | None = None,
) -> torch.Tensor:
    """
    Per-pixel weight map: pixels within `boundary_radius` of a class transition
    in `targets` get weight `boundary_weight`, all other pixels get weight 1.

    Implementation: a class transition exists at pixel p if any pixel in its
    (2r+1)×(2r+1) neighbourhood has a different class. We detect transitions by
    max-pool vs min-pool: if max != min, the patch crosses a boundary.

    Args:
        targets:         (N, H, W) int64: class indices
        boundary_weight: weight multiplier on boundary pixels (1.0 = no effect)
        boundary_radius: dilate the boundary by this many pixels (default 3)
        boundary_class:  if not None, only weight transitions that involve THIS
                         class (i.e. binary mask = (targets==boundary_class)).
                         Lets you focus the loss on, e.g., only pupil edges.
    Returns:
        (N, H, W) float weight map.
    """
    if boundary_weight <= 1.0:
        return torch.ones_like(targets, dtype=torch.float32)
    if boundary_class is not None:
        # Detect transitions of the binary "is this class?" mask only.
        t = (targets == boundary_class).float().unsqueeze(1)
    else:
        t = targets.unsqueeze(1).float()
    k = 2 * boundary_radius + 1
    max_p = F.max_pool2d(t, kernel_size=k, stride=1, padding=boundary_radius)
    min_p = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=boundary_radius)
    is_boundary = (max_p != min_p).squeeze(1).float()  # (N, H, W)
    return 1.0 + (boundary_weight - 1.0) * is_boundary

class SegmentationLoss(nn.Module):
    """CE ponderada con normalización idéntica al entrenamiento original."""
    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.tensor([0.5, 1.5, 2.0]))

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        w = _boundary_weight_map(targets, 3.0, 1, 2)
        return (ce * w).sum() / w.sum().clamp(min=1.0)
