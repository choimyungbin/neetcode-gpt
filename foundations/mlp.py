import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def relu(self, z):
        return np.maximum(0, z)
    def forward(self, 
    x: NDArray[np.float64], 
    weights: List[NDArray[np.float64]], 
    biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        z = x
        for i in range(len(weights)-1):
            # hidden layer
            z = self.relu(z @ weights[i] + biases[i])
        # output layer
        res = z @ weights[-1] + biases[-1]
        return np.round(res, 5)