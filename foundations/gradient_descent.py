class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        def derivative(x):
            return 2*x

        def end_condition(x):
            return True if derivative(x) == 0 else False

        x_old = init
        for i in range(iterations):
            x_new = x_old - learning_rate * (derivative(x_old))
            if end_condition(x_new):
                return round(x_new, 5)
            x_old = x_new
        
        return round(x_old, 5)