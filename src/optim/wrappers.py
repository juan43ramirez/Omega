"""Wrappers for the optimizers of both players. Useful for unifying the training loop"""

from torch.optim import Optimizer


class OptimizerWrapper:
    def __init__(self, x_optimizer: Optimizer, y_optimizer: Optimizer):
        self.x_optimizer = x_optimizer
        self.y_optimizer = y_optimizer

    def zero_grad(self):
        self.x_optimizer.zero_grad()
        self.y_optimizer.zero_grad()

    def step(self):
        raise NotImplementedError


class SimultaneousOptimizer(OptimizerWrapper):
    def __init__(self, x_optimizer: Optimizer, y_optimizer: Optimizer):
        super().__init__(x_optimizer, y_optimizer)

    def step(self):
        self.x_optimizer.step()

        # Flip the gradient of y to perform maximization
        self.y_optimizer.param_groups[0]["params"][0].grad *= -1
        self.y_optimizer.step()


class AlternatingOptimizer(OptimizerWrapper):
    def __init__(self, x_optimizer: Optimizer, y_optimizer: Optimizer):
        super().__init__(x_optimizer, y_optimizer)

    def step(self, closure, *closure_args, **closure_kwargs):

        # Perform a step for x
        self.x_optimizer.step()

        self.zero_grad()
        loss = closure(*closure_args, **closure_kwargs)
        loss.backward()

        # Flip the gradient of y to perform maximization
        self.y_optimizer.param_groups[0]["params"][0].grad *= -1
        self.y_optimizer.step()

        return loss


class ExtrapolationOptimizer(OptimizerWrapper):
    def __init__(self, x_optimizer: Optimizer, y_optimizer: Optimizer):
        if not hasattr(x_optimizer, "extrapolation") or not hasattr(y_optimizer, "extrapolation"):
            raise ValueError("Optimizers must implement an extrapolation method")
        super().__init__(x_optimizer, y_optimizer)

    def step(self, closure, *closure_args, **closure_kwargs):
        # Compute extrapolated iterates
        self.x_optimizer.extrapolation()

        # Flip the gradient of y to perform maximization
        self.y_optimizer.param_groups[0]["params"][0].grad *= -1
        self.y_optimizer.extrapolation()

        # Re-compute gradients at the extrapolated points
        self.zero_grad()
        loss = closure(*closure_args, **closure_kwargs)
        loss.backward()

        # Perform an update step
        self.x_optimizer.step()

        # Flip the gradient of y to perform maximization
        self.y_optimizer.param_groups[0]["params"][0].grad *= -1
        self.y_optimizer.step()

        return loss
