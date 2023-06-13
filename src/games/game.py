import math

import numpy as np
import torch

from ..utils import seed_all
from .sampler import RandomSampler
from .utils import make_random_matrix, make_sym_matrix


class StochasticQuadraticGame:
    def __init__(
        self,
        dim,
        num_samples,
        mu=0.0,
        L=1.0,
        mu_B=0.0,
        L_B=1.0,
        num_zeros=0,
        bias=False,
        make_normal_B=True,
        linear_x=False,
        linear_y=False,
        data_seed=None,
        sample_seed=None,
    ):
        super().__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.sampler = RandomSampler(num_samples, seed=sample_seed)

        # Fix the seed for the data generation for reproducibility
        seed_all(data_seed)

        # Making A and C the same shape and conditioning
        if linear_x:
            self.A = torch.zeros(num_samples, self.dim, self.dim)
        else:
            # using 2*mu and 2*L as we use A/2 in the loss
            self.A = make_sym_matrix(num_samples, self.dim, mu, L, num_zeros)

        if linear_y:
            self.C = torch.zeros(num_samples, self.dim, self.dim)
        else:
            # using 2*mu and 2*L as we use C/2 in the loss
            self.C = make_sym_matrix(num_samples, self.dim, mu, L, num_zeros)

        # B has different conditioning though
        self.B = make_random_matrix(num_samples, self.dim, mu_B, L_B, normal=make_normal_B)

        if bias:
            self.a = torch.randn(num_samples, self.dim) / math.sqrt(self.dim)
            self.c = torch.randn(num_samples, self.dim) / math.sqrt(self.dim)
        else:
            self.a = torch.zeros(num_samples, self.dim)
            self.c = torch.zeros(num_samples, self.dim)

        self.x_star, self.y_star = self.optimum()
        self.f_star = self.loss(torch.arange(num_samples), torch.tensor(self.x_star), torch.tensor(self.y_star))

        if not isinstance(self, (StochasticBilinearGame)):
            # Conditioning of the whole problem
            J = np.block([[self.A.numpy(), self.B.numpy()], [-self.B.transpose(-2, -1).numpy(), self.C.numpy()]])

            # Get the smallest and largest eigenvalues of the jacobian of the gradient field
            e = np.linalg.eigvals(J)  # for the individual matrices
            L = float((1 / ((1 / e).real.min(1))).max())
            mu = float(e.real.min())
            self.kappa = L / (mu + 1e-8)

            e_mean = np.linalg.eigvals(J.mean(0))  # for the mean matrix
            L_mean = float(1 / ((1 / e_mean).real.min()))
            mu_mean = float(e_mean.real.min())
            self.kappa_mean = L_mean / (mu_mean + 1e-8)
        else:
            self.kappa = None
            self.kappa_mean = None

        self.kappa_B = L_B / (mu_B + 1e-8)

    @property
    def num_players(self):
        return len(self.B.shape)

    def sample(self, return_index=False):
        return self.sampler.sample(return_index)

    @torch.no_grad()
    def dist2opt(self, x, y):
        dist_sq = (x.data.cpu() - self.x_star).pow(2).sum() + (y.data.cpu() - self.y_star).pow(2).sum()
        return torch.sqrt(dist_sq)

    def get_mini_batch(self, idx: list[int]):
        A = self.A[idx, ...].mean(0)
        B = self.B[idx, ...].mean(0)
        C = self.C[idx, ...].mean(0)

        a = self.a[idx, ...].mean(0)
        c = self.c[idx, ...].mean(0)

        return A, B, C, a, c

    def loss(self, idx: list[int], x, y):
        """Compute the loss of the game for minibatch idx."""

        A, B, C, a, c = self.get_mini_batch(idx)

        # xAx/2 + xBy - yCy/2 + ax - cy
        return (x @ A @ x) / 2 + x @ B @ y - (y @ C @ y) / 2 + a @ x - c @ y

    def optimum(self):
        # Get full batch matrices
        A, B, C, a, c = self.get_mini_batch(torch.arange(self.num_samples))
        A, B, C, a, c = A.numpy(), B.numpy(), C.numpy(), a.numpy(), c.numpy()

        nabla_F = np.block([[A, B], [-B.T, C]])

        bias = np.concatenate([a, c], axis=-1)
        sol = np.linalg.lstsq(nabla_F, -bias)[0]
        x_star, y_star = np.split(sol, 2)

        return x_star, y_star

    def to_(self, *args, **kwargs):
        self.A = self.A.to(*args, **kwargs)
        self.B = self.B.to(*args, **kwargs)
        self.C = self.C.to(*args, **kwargs)

        self.a = self.a.to(*args, **kwargs)
        self.c = self.c.to(*args, **kwargs)

    def state_dict(self):
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "a": self.a,
            "c": self.c,
            "x_star": self.x_star,
            "y_star": self.y_star,
            "f_star": self.f_star,
            "kappa": self.kappa,
            "kappa_mean": self.kappa_mean,
            "kappa_B": self.kappa_B,
        }

    def load_state_dict(self, state_dict):
        self.A = state_dict["A"]
        self.B = state_dict["B"]
        self.C = state_dict["C"]
        self.a = state_dict["a"]
        self.c = state_dict["c"]
        self.x_star = state_dict["x_star"]
        self.y_star = state_dict["y_star"]
        self.f_star = state_dict["f_star"]
        self.kappa = state_dict["kappa"]
        self.kappa_mean = state_dict["kappa_mean"]
        self.kappa_B = state_dict["kappa_B"]

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(dim={self.dim}, num_samples={self.num_samples}), kappa={self.kappa}, kappa_b={self.kappa_B}, f_star={self.f_star}, x_star={self.x_star}, y_star={self.y_star})"


class StochasticBilinearGame(StochasticQuadraticGame):
    def __init__(self, dim, num_samples=1, **kwargs):
        assert "linear_x" not in kwargs and "linear_y" not in kwargs
        super().__init__(dim, num_samples=num_samples, linear_x=True, linear_y=True, **kwargs)


class StochasticQuadraticLinearGame(StochasticQuadraticGame):
    def __init__(self, dim, num_samples=1, **kwargs):
        assert "linear_x" not in kwargs and "linear_y" not in kwargs
        super().__init__(dim, num_samples=num_samples, linear_x=False, linear_y=True, **kwargs)
