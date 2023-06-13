"""
Code *inspired* on https://github.com/GauthierGidel/Variational-Inequality-GAN
See A Variational Inequality Approach to Generative Adversarial Networks by
Gauthier Gidel, Hugo Berard, Gaëtan Vignoud, Pascal Vincent and Simon Lacoste-Julien
in ICLR, 2018
"""

import torch

required = object()


class OMEGA(torch.optim.Optimizer):
    def __init__(self, params, lr=required, optimism=None, ema_beta=0.9):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)

        self.optimism = optimism if optimism is not None else lr
        self.ema_beta = ema_beta

        super(OMEGA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OMEGA, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["grad_ema"] = torch.clone(d_p)

                direction = d_p + self.optimism * (d_p - state["grad_ema"])
                p.data.add_(-group["lr"] * direction)

                # Update the EMA
                state["grad_ema"] = self.ema_beta * state["grad_ema"] + (1 - self.ema_beta) * d_p.clone()

        return loss


class OMEGAM(torch.optim.Optimizer):
    """Omega with Momentum. This formulation is equivalent to that of OMEGA for specific
    choices of the optimism parameter. Included for convenience."""

    def __init__(self, params, lr=required, optimism=None, ema_beta=0.9):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)

        self.optimism = optimism if optimism is not None else lr
        self.ema_beta = ema_beta

        super(OMEGAM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OMEGAM, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["grad_ema"] = torch.clone(d_p)

                old_grad_ema = state["grad_ema"]
                new_grad_ema = self.ema_beta * state["grad_ema"] + (1 - self.ema_beta) * d_p

                direction = new_grad_ema + self.optimism * (new_grad_ema - old_grad_ema)
                p.data.add_(-group["lr"] * direction)

                # Update the EMA
                state["grad_ema"] = new_grad_ema.clone()

        return loss
