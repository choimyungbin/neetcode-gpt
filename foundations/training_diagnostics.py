import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = x.mean()
                    std = x.std()
                    dead = x <= 0
                    dead_fraction = dead.all(dim=0).float().mean()
                    res.append(
                        {'mean': round(mean.item(), 4),
                        'std': round(std.item(), 4),
                        'dead_fraction': round(dead_fraction.item(), 4)})
        return res
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        res = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = grad.mean()
                std = grad.std()
                norm = torch.norm(grad)
                res.append(
                        {'mean': round(mean.item(), 4),
                        'std': round(std.item(), 4),
                        'norm': round(norm.item(), 4)})
        return res

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        # 1. dead neurons
        if any(layer['dead_fraction'] > 0.5 for layer in activation_stats):
            return 'dead_neurons'

        # 2. exploding gradients
        if any(layer['norm'] > 1000 for layer in gradient_stats):
            return 'exploding_gradients'

        # 3. vanishing gradients
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        if any(layer['std'] < 0.1 for layer in activation_stats):
            return 'vanishing_gradients'

        # 4. exploding gradients again
        if any(layer['std'] > 10.0 for layer in activation_stats):
            return 'exploding_gradients'

        return 'healthy'