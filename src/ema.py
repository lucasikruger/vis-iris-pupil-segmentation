import torch
import torch.nn as nn
class EMA:
    """Exponential moving average of model parameters.

    Decay 0.999 ≈ effective averaging window of ~1000 steps. Updated after
    each optimizer step. evaluation/inference uses the EMA copy of the weights.
    Almost always +0.5-1.5 pp IoU for free at no extra training cost.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters() if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """Returns a dict of the model's current params so they can be restored.
        Overwrites model params with EMA values for evaluation."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
