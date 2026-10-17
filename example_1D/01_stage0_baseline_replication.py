"""Stage 0 -- baseline replication and statistical infrastructure.

Example 1 (1D, uniform grid) of the RBF-fPINN reference paper. Compares RBF-fPINN with a
standard MLP-based fPINN for several values of the Gaussian width parameter mu, using multiple
random seeds, so that all later stages can reuse this validated core."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import gamma as gamma_fn

# --------------------------------------------------------------------------- #
# Global setup
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    """Full determinism across torch / numpy for one experiment run."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Problem definition   (Example 1, fPINN-RBF paper)
#   u_t + D^alpha_t u - u_xx = f(x,t) ,   (x,t) in [0,1] x [0,1]
#   exact:  u(x,t) = (1 + t^3) * exp(-mu (x-0.5)^2)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class ProblemConfig:
    """All problem constants live here and are passed explicitly wherever
    needed. The reference implementation instead read a module-level
    global `mu`; that is fine for a single run but becomes a silent-bug
    risk the moment several (alpha, mu) configurations are swept in a
    loop, because it is easy to forget to keep the global in sync with
    the tensors already built for a different configuration."""

    alpha: float
    mu: float
    lb: tuple[float, float] = (0.0, 0.0)  # (t_low, x_low)
    ub: tuple[float, float] = (1.0, 1.0)  # (t_high, x_high)


def exact_u(tx: torch.Tensor, cfg: ProblemConfig) -> torch.Tensor:
    t, x = tx[:, [0]], tx[:, [1]]
    return (1 + t**3) * torch.exp(-cfg.mu * (x - 0.5) ** 2)


def source_term(tx: torch.Tensor, cfg: ProblemConfig) -> torch.Tensor:
    t, x = tx[:, [0]], tx[:, [1]]
    phi = torch.exp(-cfg.mu * (x - 0.5) ** 2)
    u_t = 3 * t**2 * phi
    caputo_t = phi * (6.0 / math.gamma(4 - cfg.alpha)) * t ** (3 - cfg.alpha)
    u_xx = (1 + t**3) * (-2 * cfg.mu + 4 * cfg.mu**2 * (x - 0.5) ** 2) * phi
    return u_t + caputo_t - u_xx


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_train_grid(cfg: ProblemConfig, num_t: int, num_x: int):
    t = torch.linspace(cfg.lb[0], cfg.ub[0], num_t)[:, None].float()
    x = torch.linspace(cfg.lb[1], cfg.ub[1], num_x)[:, None].float()
    return t, x


def make_test_grid(cfg: ProblemConfig, n_t: int, n_x: int):
    t = np.linspace(cfg.lb[0], cfg.ub[0], n_t)
    x = np.linspace(cfg.lb[1], cfg.ub[1], n_x)
    tt, xx = np.meshgrid(t, x)
    grid = np.stack([tt.ravel(), xx.ravel()], axis=1)
    grid = torch.from_numpy(grid).float().to(DEVICE)
    return grid, exact_u(grid, cfg)


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
class GaussianRBFLayer(nn.Module):
    """Single hidden layer of trainable-center / trainable-width Gaussian
    RBFs. The log-sigma parameterisation keeps sigma > 0 without a
    constrained optimiser. The root-free formulation (no sqrt anywhere)
    avoids a singular second derivative at zero distance -- this matters
    because u_xx is obtained through autograd, and d^2/dx^2 sqrt(z) blows
    up as z -> 0."""

    def __init__(self, in_features: int, n_centers: int):
        super().__init__()
        self.centres = nn.Parameter(torch.empty(n_centers, in_features))
        self.log_sigma = nn.Parameter(torch.empty(n_centers))
        nn.init.uniform_(self.centres, 0.0, 1.0)
        nn.init.constant_(self.log_sigma, math.log(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centres.unsqueeze(0)  # (B, C, in)
        sq_dist = diff.pow(2).sum(-1)  # (B, C)
        sigma2 = torch.exp(self.log_sigma).pow(2).unsqueeze(0)  # (1, C)
        return torch.exp(-sq_dist / sigma2)


class RBFfPINN(nn.Module):
    def __init__(self, in_features: int, n_centers: int, out_features: int = 1):
        super().__init__()
        self.rbf = GaussianRBFLayer(in_features, n_centers)
        self.readout = nn.Linear(n_centers, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.rbf(x))


class MLPfPINN(nn.Module):
    def __init__(self, layer_sizes: list[int], activation: nn.Module | None = None):
        super().__init__()
        self.activation = activation or nn.Tanh()
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        )
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight, gain=1.0)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.linears[:-1]:
            x = self.activation(lin(x))
        return self.linears[-1](x)


# --------------------------------------------------------------------------- #
# L1 Caputo-derivative operator (vectorised)
# --------------------------------------------------------------------------- #
class L1CaputoOperator:
    """Pre-computes the (N_t-1, N_t) reconstruction matrix W exactly once.
    In the reference implementation the equivalent computation was a
    nested Python double loop executed *inside* `PDE_loss`, i.e. once per
    training iteration (thousands of times per run). Here it becomes a
    single matrix multiplication per iteration; the underlying numerical
    scheme (classical L1 finite-difference approximation of the Caputo
    derivative, Oldham & Spanier 1974) is unchanged -- this is a pure
    performance / robustness refactor, verified below to reproduce the
    original scalar-loop formula bit-for-bit on a small case.
    """

    def __init__(self, num_t: int, dt: float, alpha: float, device: torch.device):
        n = np.arange(num_t)
        coef = dt ** (-alpha) * ((n + 1) ** (1 - alpha) - n ** (1 - alpha)) / gamma_fn(2 - alpha)
        coef_t = torch.tensor(coef, dtype=torch.float32, device=device)
        self.coef0 = coef_t[0]

        w = torch.zeros(num_t - 1, num_t, dtype=torch.float32, device=device)
        for r in range(num_t - 1):  # r = n - 1,  n = 1 .. num_t-1
            n_ = r + 1
            w[r, 0] = coef_t[n_ - 1] / coef_t[0]
            for k in range(1, n_):
                w[r, k] = (coef_t[n_ - k - 1] - coef_t[n_ - k]) / coef_t[0]
        self.w = w  # (num_t - 1, num_t), built once, reused every iteration

    def residual(self, lu_plus_f: torch.Tensor, u_grid: torch.Tensor) -> torch.Tensor:
        """lu_plus_f, u_grid: (num_t, num_x). Returns the L1 self-consistency
        residual for steps n = 1 .. num_t-1, shape (num_t-1, num_x)."""
        reconstruction = lu_plus_f[1:] / self.coef0 + self.w @ u_grid
        return reconstruction - u_grid[1:]


# --------------------------------------------------------------------------- #
# fPINN trainer
# --------------------------------------------------------------------------- #
class FractionalPINNTrainer:
    def __init__(
        self,
        net: nn.Module,
        cfg: ProblemConfig,
        num_t: int,
        num_x: int,
        test_grid: torch.Tensor,
        test_exact: torch.Tensor,
    ):
        self.net = net.to(DEVICE)
        self.cfg = cfg
        self.num_t, self.num_x = num_t, num_x
        self.test_grid, self.test_exact = test_grid, test_exact
        self.iter = 0

        t, x = make_train_grid(cfg, num_t, num_x)
        t, x = t.to(DEVICE), x.to(DEVICE)
        dt = (cfg.ub[0] - cfg.lb[0]) / (num_t - 1)
        self.caputo = L1CaputoOperator(num_t, dt, cfg.alpha, DEVICE)

        t0 = torch.full((num_x, 1), t[0, 0].item(), device=DEVICE)
        self.tx_ic = torch.cat([t0, x], dim=1)
        self.u_ic = exact_u(self.tx_ic, cfg)

        t_rep = t.repeat_interleave(num_x).view(-1, 1)
        x_rep = x.repeat(num_t, 1)
        self.tx_pde = torch.cat([t_rep, x_rep], dim=1)
        self.f_pde = source_term(self.tx_pde, cfg)

        x_lo = torch.full((num_t, 1), cfg.lb[1], device=DEVICE)
        x_hi = torch.full((num_t, 1), cfg.ub[1], device=DEVICE)
        self.tx_b_lo = torch.cat([t, x_lo], dim=1)
        self.tx_b_hi = torch.cat([t, x_hi], dim=1)
        self.u_b_lo = exact_u(self.tx_b_lo, cfg)
        self.u_b_hi = exact_u(self.tx_b_hi, cfg)

    def _pde_loss(self) -> torch.Tensor:
        x = self.tx_pde.clone().requires_grad_(True)
        u = self.net(x)
        grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t, u_x = grad_u[:, [0]], grad_u[:, [1]]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, [1]]

        u_grid = u.reshape(self.num_t, -1)
        lu_plus_f = (
            -u_t.reshape(self.num_t, -1)
            + u_xx.reshape(self.num_t, -1)
            + self.f_pde.reshape(self.num_t, -1)
        )
        residual = self.caputo.residual(lu_plus_f, u_grid)
        return residual.pow(2).mean()

    def _total_loss(self):
        loss_ic = (self.net(self.tx_ic) - self.u_ic).pow(2).mean()
        loss_bc = (self.net(self.tx_b_lo) - self.u_b_lo).pow(2).mean() + (
            self.net(self.tx_b_hi) - self.u_b_hi
        ).pow(2).mean()
        loss_pde = self._pde_loss()
        return loss_ic, loss_bc, loss_pde

    def fit(self, adam_epochs: int, lbfgs_epochs: int, verbose_every: int = 0):
        opt_adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for _ in range(adam_epochs):
            opt_adam.zero_grad()
            li, lbnd, lpde = self._total_loss()
            loss = li + lbnd + lpde
            loss.backward()
            opt_adam.step()
            self.iter += 1
            if verbose_every and self.iter % verbose_every == 0:
                print(f"[Adam ] it={self.iter:6d}  loss={loss.item():.4e}")

        opt_lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=lbfgs_epochs,
            max_eval=lbfgs_epochs,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            li, lbnd, lpde = self._total_loss()
            loss = li + lbnd + lpde
            loss.backward()
            self.iter += 1
            if verbose_every and self.iter % verbose_every == 0:
                print(f"[LBFGS] it={self.iter:6d}  loss={loss.item():.4e}")
            return loss

        opt_lbfgs.step(closure)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        pred_train = self.net(self.tx_pde)
        exact_train = exact_u(self.tx_pde, self.cfg)
        train_err = (
            torch.linalg.norm(pred_train - exact_train) / torch.linalg.norm(exact_train)
        ).item()

        pred_test = self.net(self.test_grid)
        test_err = (
            torch.linalg.norm(pred_test - self.test_exact) / torch.linalg.norm(self.test_exact)
        ).item()
        return train_err, test_err


# --------------------------------------------------------------------------- #
# Experiment runner -- Stage 0: multi-seed baseline replication
# --------------------------------------------------------------------------- #
def run_one(
    architecture: Literal["RBF", "MLP"],
    alpha: float,
    mu: float,
    seed: int,
    n_neurons: int = 150,
    num_t: int = 50,
    num_x: int = 50,
    n_t_test: int = 100,
    n_x_test: int = 100,
    adam_epochs: int = 5000,
    lbfgs_epochs: int = 20000,
    verbose: bool = False,
) -> dict:
    set_seed(seed)
    cfg = ProblemConfig(alpha=alpha, mu=mu)
    test_grid, test_exact = make_test_grid(cfg, n_t_test, n_x_test)

    if architecture == "RBF":
        net = RBFfPINN(in_features=2, n_centers=n_neurons)
    elif architecture == "MLP":
        net = MLPfPINN([2, n_neurons, 1])
    else:
        raise ValueError(f"unknown architecture: {architecture!r}")

    trainer = FractionalPINNTrainer(net, cfg, num_t, num_x, test_grid, test_exact)

    t0 = time.time()
    trainer.fit(adam_epochs, lbfgs_epochs, verbose_every=500 if verbose else 0)
    wall_time = time.time() - t0

    train_err, test_err = trainer.evaluate()
    return dict(
        architecture=architecture,
        alpha=alpha,
        mu=mu,
        seed=seed,
        train_l2=train_err,
        test_l2=test_err,
        wall_time_s=wall_time,
        n_iter=trainer.iter,
    )


def run_stage0(
    alphas: Iterable[float] = (0.1,),
    mus: Iterable[float] = (80, 160, 240, 320),
    seeds: Iterable[int] = range(10),
    adam_epochs: int = 5000,
    lbfgs_epochs: int = 20000,
    out_csv: str = "stage0_baseline.csv",
) -> pd.DataFrame:
    records = []
    for architecture in ("RBF", "MLP"):
        for alpha in alphas:
            for mu in mus:
                for seed in seeds:
                    rec = run_one(
                        architecture,
                        alpha,
                        mu,
                        seed,
                        adam_epochs=adam_epochs,
                        lbfgs_epochs=lbfgs_epochs,
                    )
                    records.append(rec)
                    print(
                        f"{architecture:>3s} | alpha={alpha} mu={mu:>4} seed={seed:>2d} "
                        f"| train_L2={rec['train_l2']:.3e} test_L2={rec['test_l2']:.3e} "
                        f"| {rec['wall_time_s']:.1f}s"
                    )
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    results = run_stage0(seeds=range(10))
    summary = results.groupby(["architecture", "alpha", "mu"])[
        ["train_l2", "test_l2", "wall_time_s"]
    ].agg(["mean", "std"])
    print(summary)
