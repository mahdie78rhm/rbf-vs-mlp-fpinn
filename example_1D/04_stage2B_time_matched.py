"""Stage 2, Scenario B -- time-matched comparison (self-contained and resumable).

For each (mu, seed):
  1. Train the RBF-fPINN once with 150 centers and record its wall-clock time T.
  2. Train the Stage-1 winning MLP (2 hidden layers x 50, tanh) with independent restart seeds for
     up to the same wall-clock budget T (at least one restart always runs); keep the restart with
     the lowest final TRAINING loss (never the test loss, to avoid test-set leakage).
  3. Report the test error of both architectures for this seed-matched pair."""

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
# 0. Core definitions (duplicated on purpose -- every stage is self-contained)
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

    def fit(self, adam_epochs: int, lbfgs_epochs: int, verbose_every: int = 0) -> float:
        """Returns the FINAL TRAINING LOSS (sum of ic+bc+pde) after fitting --
        used for restart selection in the time-matched MLP protocol below."""
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
        final_loss_holder = {"value": None}

        def closure():
            opt_lbfgs.zero_grad()
            li, lbnd, lpde = self._total_loss()
            loss = li + lbnd + lpde
            loss.backward()
            self.iter += 1
            final_loss_holder["value"] = loss.item()
            if verbose_every and self.iter % verbose_every == 0:
                print(f"[LBFGS] it={self.iter:6d}  loss={loss.item():.4e}")
            return loss

        opt_lbfgs.step(closure)
        return final_loss_holder["value"]

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
# 1. Pre-registered seed pool (identical across ALL stages/notebooks/accounts)
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

# Stage-1 winning MLP configuration (2 hidden layers x 50, tanh)
MLP_HIDDEN_SIZES = [50, 50]
RBF_N_CENTERS = 150  # the paper's original setting, matching Stage 0


# =========================================================================== #
# 2. One RBF run (reference) + time-matched MLP multi-restart, for one
#    (mu, seed) pair
# =========================================================================== #
def _train_once(
    net: nn.Module,
    cfg: ProblemConfig,
    num_t: int,
    num_x: int,
    test_grid,
    test_exact,
    adam_epochs: int,
    lbfgs_epochs: int,
):
    trainer = FractionalPINNTrainer(net, cfg, num_t, num_x, test_grid, test_exact)
    t0 = time.time()
    final_train_loss = trainer.fit(adam_epochs, lbfgs_epochs, verbose_every=0)
    wall_time = time.time() - t0
    train_l2, test_l2 = trainer.evaluate()
    return dict(
        final_train_loss=final_train_loss, wall_time=wall_time, train_l2=train_l2, test_l2=test_l2
    )


def run_time_matched_pair(
    mu: float,
    seed: int,
    alpha: float = 0.1,
    num_t: int = 50,
    num_x: int = 50,
    n_t_test: int = 100,
    n_x_test: int = 100,
    adam_epochs: int = 5000,
    lbfgs_epochs: int = 20000,
    max_restarts: int = 8,
) -> dict:
    cfg = ProblemConfig(alpha=alpha, mu=mu)
    test_grid, test_exact = make_test_grid(cfg, n_t_test, n_x_test)

    # --- 1. reference RBF run: measures the time budget for this pair ---
    set_seed(seed)
    rbf_net = RBFfPINN(in_features=2, n_centers=RBF_N_CENTERS)
    rbf_result = _train_once(
        rbf_net, cfg, num_t, num_x, test_grid, test_exact, adam_epochs, lbfgs_epochs
    )
    time_budget = rbf_result["wall_time"]

    # --- 2. time-matched MLP: restart with fresh seeds until the RBF's
    #        wall-clock budget is spent (always run at least 1 restart) ---
    restart_seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(max_restarts)]
    mlp_restarts = []
    cumulative_time = 0.0
    for r_seed in restart_seeds:
        if mlp_restarts and cumulative_time >= time_budget:
            break
        set_seed(r_seed)
        mlp_net = MLPfPINN([2, *MLP_HIDDEN_SIZES, 1])
        result = _train_once(
            mlp_net, cfg, num_t, num_x, test_grid, test_exact, adam_epochs, lbfgs_epochs
        )
        cumulative_time += result["wall_time"]
        mlp_restarts.append(result)

    # model selection by TRAINING loss only -- never by test_l2 (no leakage)
    best = min(mlp_restarts, key=lambda r: r["final_train_loss"])

    return dict(
        mu=mu,
        seed=seed,
        rbf_test_l2=rbf_result["test_l2"],
        rbf_train_l2=rbf_result["train_l2"],
        rbf_wall_time_s=rbf_result["wall_time"],
        mlp_test_l2=best["test_l2"],
        mlp_train_l2=best["train_l2"],
        mlp_n_restarts=len(mlp_restarts),
        mlp_wall_time_total_s=cumulative_time,
        mlp_best_restart_final_train_loss=best["final_train_loss"],
    )


# =========================================================================== #
# 3. Resumable sweep (identical file-based resume pattern as every stage)
# =========================================================================== #
_KEY_COLS = ["mu", "seed"]


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


def _print_existing_results(csv_path: str) -> None:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("(no previous rows to show)")
        return
    existing = pd.read_csv(csv_path)
    print(f"\n--- {len(existing)} previously completed rows ---")
    for _, row in existing.iterrows():
        print(
            f"[DONE] mu={row.mu:>3} seed={int(row.seed):>10d} | "
            f"RBF test_L2={row.rbf_test_l2:.3e} ({row.rbf_wall_time_s:.0f}s) | "
            f"MLP(time-matched, {int(row.mlp_n_restarts)} restarts) test_L2={row.mlp_test_l2:.3e} "
            f"({row.mlp_wall_time_total_s:.0f}s)"
        )
    print(f"--- end of {len(existing)} previous rows, continuing with what's left ---\n")


def run_time_matched_sweep(
    mus: Iterable[float], seeds: Iterable[int], csv_path: str
) -> pd.DataFrame:
    done = _load_done_keys(csv_path)
    mus, seeds = list(mus), list(seeds)
    total = len(mus) * len(seeds)
    n_run = n_skipped = 0

    for mu in mus:
        for seed in seeds:
            key = (mu, seed)
            if key in done:
                n_skipped += 1
                continue
            rec = run_time_matched_pair(mu, seed)
            _append_row(rec, csv_path)
            n_run += 1
            print(
                f"[TIME-MATCH] mu={mu:>3} seed={seed:>10d} | "
                f"RBF test_L2={rec['rbf_test_l2']:.3e} ({rec['rbf_wall_time_s']:.0f}s) | "
                f"MLP({rec['mlp_n_restarts']} restarts) test_L2={rec['mlp_test_l2']:.3e} "
                f"({rec['mlp_wall_time_total_s']:.0f}s) "
                f"| ({n_run + n_skipped}/{total}, {n_skipped} resumed)"
            )

    return pd.read_csv(csv_path)


# =========================================================================== #
# 4. Run
# =========================================================================== #
MUS = [80, 160, 240, 320]
CSV_PATH = "stage2_time_matched.csv"

# resume from a previously-uploaded checkpoint dataset, if present
import shutil, glob

_candidates = glob.glob(
    os.path.join(PREV_RESULTS_DIR, "**", "stage2_time_matched.csv"), recursive=True
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

_print_existing_results(CSV_PATH)

result_df = run_time_matched_sweep(MUS, SEED_POOL, CSV_PATH)

print("\n=== Time-matched comparison: summary ===")
print(
    result_df.groupby("mu")[
        ["rbf_test_l2", "mlp_test_l2", "rbf_wall_time_s", "mlp_wall_time_total_s", "mlp_n_restarts"]
    ].agg(["mean", "std"])
)
