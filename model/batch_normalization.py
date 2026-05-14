import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.array(x)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        # During training: normalize using batch statistics, then update running stats
        if training == True:
            means = np.mean(x, axis=0)
            var = np.sum((x - means)**2, axis=0)/len(x)
            x_hat = (x - means)/np.sqrt(var + eps)
            running_mean = (1.0 - momentum) * running_mean + momentum * means
            running_var = (1.0 - momentum) * running_var + momentum * var
            
        # During inference: normalize using running stats (no batch stats needed)
        if training == False:
            x_hat = (x - running_mean)/np.sqrt(running_var+eps)
            
        # Apply affine transform: y = gamma * x_hat + beta
        y = x_hat * gamma + beta
        
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4))
