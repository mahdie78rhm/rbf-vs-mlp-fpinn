"""Stage 1 -- fair tuning of the MLP-fPINN (self-contained and resumable).

Phase A screens depth x width x activation on mu in {80, 320}; Phase B refines the best
configurations on the full mu grid with all seeds. Results are appended to
stage1_coarse_search.csv / stage1_refine_results.csv; on restart, rows already present are
skipped, so an interrupted run loses at most the run in progress."""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import gamma as gamma_fn

# =========================================================================== #
# 0. Core definitions (identical to Stage 0 -- duplicated here on purpose so
#    this file has zero cross-cell dependency; see note at top of file).
# =========================================================================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@dataclasses.dataclass(frozen=True)
class ProblemConfig:
    alpha: float
    mu: float
    lb: tuple[float, float] = (0.0, 0.0)
    ub: tuple[float, float] = (1.0, 1.0)


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


class GaussianRBFLayer(nn.Module):
    def __init__(self, in_features: int, n_centers: int):
        super().__init__()
        self.centres = nn.Parameter(torch.empty(n_centers, in_features))
        self.log_sigma = nn.Parameter(torch.empty(n_centers))
        nn.init.uniform_(self.centres, 0.0, 1.0)
        nn.init.constant_(self.log_sigma, math.log(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centres.unsqueeze(0)
        sq_dist = diff.pow(2).sum(-1)
        sigma2 = torch.exp(self.log_sigma).pow(2).unsqueeze(0)
        return torch.exp(-sq_dist / sigma2)


class RBFfPINN(nn.Module):
    def __init__(self, in_features: int, n_centers: int, out_features: int = 1):
        super().__init__()
        self.rbf = GaussianRBFLayer(in_features, n_centers)
        self.readout = nn.Linear(n_centers, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.rbf(x))


class Sin(nn.Module):
    """sin activation -- reported in the fPINN literature to help with
    oscillatory / sharply-peaked target functions such as this problem's
    narrow Gaussian profile."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


_ACTIVATIONS = {"tanh": nn.Tanh, "sin": Sin, "gelu": nn.GELU, "swish": nn.SiLU}


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


def build_mlp(hidden_sizes: list[int], activation_name: str) -> MLPfPINN:
    layer_sizes = [2, *hidden_sizes, 1]
    return MLPfPINN(layer_sizes, activation=_ACTIVATIONS[activation_name]())


class L1CaputoOperator:
    def __init__(self, num_t: int, dt: float, alpha: float, device: torch.device):
        n = np.arange(num_t)
        coef = dt ** (-alpha) * ((n + 1) ** (1 - alpha) - n ** (1 - alpha)) / gamma_fn(2 - alpha)
        coef_t = torch.tensor(coef, dtype=torch.float32, device=device)
        self.coef0 = coef_t[0]
        w = torch.zeros(num_t - 1, num_t, dtype=torch.float32, device=device)
        for r in range(num_t - 1):
            n_ = r + 1
            w[r, 0] = coef_t[n_ - 1] / coef_t[0]
            for k in range(1, n_):
                w[r, k] = (coef_t[n_ - k - 1] - coef_t[n_ - k]) / coef_t[0]
        self.w = w

    def residual(self, lu_plus_f: torch.Tensor, u_grid: torch.Tensor) -> torch.Tensor:
        reconstruction = lu_plus_f[1:] / self.coef0 + self.w @ u_grid
        return reconstruction - u_grid[1:]


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


# =========================================================================== #
# 1. Pre-registered seed pool (fixed once, never regenerated at runtime)
# =========================================================================== #
MASTER_SEED = 20250826
SEED_POOL = [
    908174675,
    2021242550,
    1298657243,
    2554806439,
    3639288809,
    1404135774,
    1953773272,
    886292835,
    2767164605,
    3922647745,
]
assert SEED_POOL == [int(s) for s in np.random.SeedSequence(MASTER_SEED).generate_state(10)]


# =========================================================================== #
# 2. Single-run wrapper for an MLP configuration
# =========================================================================== #
def run_mlp_config(
    hidden_sizes: list[int],
    activation_name: str,
    alpha: float,
    mu: float,
    seed: int,
    num_t: int = 50,
    num_x: int = 50,
    n_t_test: int = 100,
    n_x_test: int = 100,
    adam_epochs: int = 5000,
    lbfgs_epochs: int = 20000,
) -> dict:
    set_seed(seed)
    cfg = ProblemConfig(alpha=alpha, mu=mu)
    test_grid, test_exact = make_test_grid(cfg, n_t_test, n_x_test)
    net = build_mlp(hidden_sizes, activation_name)
    trainer = FractionalPINNTrainer(net, cfg, num_t, num_x, test_grid, test_exact)

    t0 = time.time()
    trainer.fit(adam_epochs, lbfgs_epochs, verbose_every=0)
    wall_time = time.time() - t0

    train_err, test_err = trainer.evaluate()
    n_params = sum(p.numel() for p in net.parameters())
    return dict(
        depth=len(hidden_sizes),
        width=hidden_sizes[0],
        activation=activation_name,
        alpha=alpha,
        mu=mu,
        seed=seed,
        train_l2=train_err,
        test_l2=test_err,
        wall_time_s=wall_time,
        n_params=n_params,
    )


# =========================================================================== #
# 3. Resumable runner: appends to disk immediately, skips completed rows
# =========================================================================== #
_KEY_COLS = ["depth", "width", "activation", "mu", "seed"]


def _load_done_keys(csv_path: str) -> set[tuple]:
    """Returns the set of (depth, width, activation, mu, seed) tuples already
    present on disk -- correct across process restarts because it is derived
    from the FILE, never from an in-memory flag."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p)
    return set(df[_KEY_COLS].itertuples(index=False, name=None))


def _append_row(record: dict, csv_path: str) -> None:
    """Append one row; writes the header iff the file does not yet exist
    (or is empty) -- safe to call from a brand-new process at any time."""
    p = Path(csv_path)
    write_header = (not p.exists()) or p.stat().st_size == 0
    pd.DataFrame([record]).to_csv(p, mode="a", index=False, header=write_header)


def run_sweep(
    configs: list[dict],
    mus: Iterable[float],
    seeds: Iterable[int],
    csv_path: str,
    tag: str,
    **fit_kwargs,
) -> pd.DataFrame:
    """configs: list of {"depth":.., "width":.., "activation":..} dicts.
    Runs every (config, mu, seed) combination not already present in
    csv_path, appending as it goes. Safe to interrupt and re-run."""
    done = _load_done_keys(csv_path)
    total = len(configs) * len(list(mus)) * len(list(seeds))
    mus, seeds = list(mus), list(seeds)  # re-materialise after the len() consumed iterators above
    n_skipped = 0
    n_run = 0

    for cfg in configs:
        hidden_sizes = [cfg["width"]] * cfg["depth"]
        for mu in mus:
            for seed in seeds:
                key = (cfg["depth"], cfg["width"], cfg["activation"], mu, seed)
                if key in done:
                    n_skipped += 1
                    continue
                rec = run_mlp_config(
                    hidden_sizes, cfg["activation"], alpha=0.1, mu=mu, seed=seed, **fit_kwargs
                )
                _append_row(rec, csv_path)
                n_run += 1
                print(
                    f"[{tag}] depth={cfg['depth']} width={cfg['width']:>3} "
                    f"act={cfg['activation']:>5} mu={mu:>3} seed={seed:>10d} "
                    f"| test_L2={rec['test_l2']:.3e} | {rec['wall_time_s']:.1f}s "
                    f"| ({n_run + n_skipped}/{total}, {n_skipped} resumed)"
                )

    return pd.read_csv(csv_path)


# =========================================================================== #
# 4. Phase A (coarse) then Phase B (refine) -- run this cell as many times as
#    needed; each call only does the work not yet done.
# =========================================================================== #
DEPTHS = [1, 2, 3, 4]
WIDTHS = [50, 100, 150, 200]
ACTIVATIONS = ["tanh", "sin"]
COARSE_MUS = [80, 320]
COARSE_SEEDS = SEED_POOL[:3]
COARSE_CSV = "stage1_coarse_search.csv"
REFINE_CSV = "stage1_refine_results.csv"

coarse_configs = [
    {"depth": d, "width": w, "activation": a}
    for d, w, a in itertools.product(DEPTHS, WIDTHS, ACTIVATIONS)
]

print("=== Phase A: coarse search ===")
coarse_df = run_sweep(coarse_configs, COARSE_MUS, COARSE_SEEDS, COARSE_CSV, tag="COARSE")

ranking = (
    coarse_df.groupby(["depth", "width", "activation"])["test_l2"]
    .mean()
    .sort_values()
    .reset_index()
)
print("\nCoarse-search ranking (best first):")
print(ranking.head(10).to_string(index=False))

top_configs = ranking.head(3)[["depth", "width", "activation"]].to_dict("records")

print("\n=== Phase B: refine top-3 configs on the full mu/seed grid ===")
refine_df = run_sweep(top_configs, [80, 160, 240, 320], SEED_POOL, REFINE_CSV, tag="REFINE")

final_summary = refine_df.groupby(["depth", "width", "activation", "mu"])[
    ["train_l2", "test_l2", "wall_time_s", "n_params"]
].agg(["mean", "std"])
print("\n=== Final MLP configuration ranking ===")
print(final_summary)
