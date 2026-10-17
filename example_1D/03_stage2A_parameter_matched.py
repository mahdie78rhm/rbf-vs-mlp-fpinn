"""Stage 2, Scenario A -- parameter-matched comparison (self-contained and resumable).

Stage 1 gives the MLP test error at four parameter budgets (601, 2751, 5301, 20601). This script
trains the RBF-fPINN at the matching budgets (c centers = 4c parameters: 2c for the centers, c for
log sigma and c for the bias-free readout), so both architectures are compared at the same
parameter count. Completed rows are stored in a CSV and skipped on restart."""

from __future__ import annotations

import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import gamma as gamma_fn

# earlier checkpoints are searched for under this directory
PREV_RESULTS_DIR = os.environ.get("PREV_RESULTS_DIR", "/kaggle/input")

# =========================================================================== #
# 0. Core definitions (duplicated on purpose -- see Stage 1 for rationale)
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

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rbf_centers_for_param_budget(target_params: int, in_features: int = 2) -> int:
    """RBF param count = in_features*c (centres) + c (log_sigma) + c (readout,
    no bias) = (in_features + 2) * c. Returns the integer c closest to the
    target parameter budget."""
    return max(1, round(target_params / (in_features + 2)))


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
        self._last_train_loss = None

        def closure():
            opt_lbfgs.zero_grad()
            li, lbnd, lpde = self._total_loss()
            loss = li + lbnd + lpde
            loss.backward()
            self.iter += 1
            self._last_train_loss = loss.item()
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
# 1. Pre-registered seed pool (identical to Stage 0/1)
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
# 2. Single-run wrapper for RBF at a given center count
# =========================================================================== #
def run_rbf_config(
    n_centers: int,
    alpha: float,
    mu: float,
    seed: int,
    num_t: int = 50,
    num_x: int = 50,
    n_t_test: int = 100,
    n_x_test: int = 100,
    adam_epochs: int = 5000,
    lbfgs_epochs: int = 20000,
    verbose_every: int = 1000,
) -> dict:
    set_seed(seed)
    cfg = ProblemConfig(alpha=alpha, mu=mu)
    test_grid, test_exact = make_test_grid(cfg, n_t_test, n_x_test)
    net = RBFfPINN(in_features=2, n_centers=n_centers)
    trainer = FractionalPINNTrainer(net, cfg, num_t, num_x, test_grid, test_exact)

    t0 = time.time()
    trainer.fit(adam_epochs, lbfgs_epochs, verbose_every=verbose_every)
    wall_time = time.time() - t0

    train_err, test_err = trainer.evaluate()
    return dict(
        architecture="RBF",
        n_centers=n_centers,
        target_params=None,  # filled by caller
        alpha=alpha,
        mu=mu,
        seed=seed,
        train_l2=train_err,
        test_l2=test_err,
        wall_time_s=wall_time,
        n_params=net.n_params(),
    )


# =========================================================================== #
# 3. Resumable sweep (same file-based resume pattern as Stage 1)
# =========================================================================== #
_KEY_COLS = ["target_params", "mu", "seed"]


def _load_done_keys(csv_path: str) -> set[tuple]:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p)
    return set(df[_KEY_COLS].itertuples(index=False, name=None))


def _append_row(record: dict, csv_path: str) -> None:
    p = Path(csv_path)
    write_header = (not p.exists()) or p.stat().st_size == 0
    pd.DataFrame([record]).to_csv(p, mode="a", index=False, header=write_header)


def run_param_matched_sweep(
    target_param_budgets: Iterable[int],
    mus: Iterable[float],
    seeds: Iterable[int],
    csv_path: str,
    **fit_kwargs,
) -> pd.DataFrame:
    done = _load_done_keys(csv_path)
    budgets, mus, seeds = list(target_param_budgets), list(mus), list(seeds)
    total = len(budgets) * len(mus) * len(seeds)
    n_run = n_skipped = 0

    for target in budgets:
        n_centers = rbf_centers_for_param_budget(target)
        for mu in mus:
            for seed in seeds:
                key = (target, mu, seed)
                if key in done:
                    n_skipped += 1
                    continue
                rec = run_rbf_config(n_centers, alpha=0.1, mu=mu, seed=seed, **fit_kwargs)
                rec["target_params"] = target
                _append_row(rec, csv_path)
                n_run += 1
                print(
                    f"[PARAM-MATCH] target_params={target:>6} (c={n_centers:>4}) mu={mu:>3} "
                    f"seed={seed:>10d} | test_L2={rec['test_l2']:.3e} | actual_params={rec['n_params']} "
                    f"| {rec['wall_time_s']:.1f}s | ({n_run + n_skipped}/{total}, {n_skipped} resumed)"
                )

    return pd.read_csv(csv_path)


# =========================================================================== #
# 4. Run: RBF at the 3 NEW parameter budgets from Stage 1
#    (target_params=601, i.e. c=150, is already covered by Stage-0's RBF run
#    at n_centers=150 -- no need to repeat it here)
# =========================================================================== #
TARGET_PARAM_BUDGETS = [2751, 5301, 20601]
MUS = [80, 160, 240, 320]
CSV_PATH = "stage2_param_matched_rbf.csv"

# --- resume from a previously-uploaded checkpoint dataset, if present ---
import shutil, glob

_candidates = glob.glob(
    os.path.join(PREV_RESULTS_DIR, "**", "stage2_param_matched_rbf.csv"), recursive=True
)
if _candidates and not os.path.exists(CSV_PATH):
    _checkpoint = _candidates[0]
    shutil.copy(_checkpoint, CSV_PATH)
    _n_rows = sum(1 for _ in open(_checkpoint)) - 1
    print(f"resumed from checkpoint: {_checkpoint} ({_n_rows} rows)")
elif os.path.exists(CSV_PATH):
    _n_rows = sum(1 for _ in open(CSV_PATH)) - 1
    print(f"{CSV_PATH} already exists with {_n_rows} rows, continuing with it as-is")
else:
    print("no checkpoint found under PREV_RESULTS_DIR -- starting from scratch")
# -------------------------------------------------------------------------

print("Parameter budgets -> matching RBF center counts:")
for t in TARGET_PARAM_BUDGETS:
    c = rbf_centers_for_param_budget(t)
    print(f"  target={t:>6}  ->  c={c:>4}  ->  actual RBF params={(2 + 2) * c}")


def _print_existing_results(csv_path: str) -> None:
    """Print every row already saved in csv_path, in the same format used
    for live runs, so the full history is visible in THIS session's log
    too -- not just a row count."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("(no previous rows to show)")
        return
    existing = pd.read_csv(csv_path)
    print(f"\n--- {len(existing)} previously completed rows ---")
    for _, row in existing.iterrows():
        print(
            f"[DONE] target_params={int(row.target_params):>6} (c={int(row.n_centers):>4}) "
            f"mu={row.mu:>3} seed={int(row.seed):>10d} | test_L2={row.test_l2:.3e} "
            f"| actual_params={int(row.n_params)} | {row.wall_time_s:.1f}s"
        )
    print(f"--- end of {len(existing)} previous rows, continuing with what's left ---\n")


_print_existing_results(CSV_PATH)

result_df = run_param_matched_sweep(TARGET_PARAM_BUDGETS, MUS, SEED_POOL, CSV_PATH)

print("\n=== RBF at matched parameter budgets: summary ===")
print(result_df.groupby(["target_params", "mu"])[["test_l2", "wall_time_s"]].agg(["mean", "std"]))
