"""Example 4.2 (2D) - Stage 0: multi-seed reproduction of the RBF-fPINN results.

============================================================================
Example 4.2 (butterfly & pentagram) -- exact reproduction of the RBF-fPINN
paper results. Single self-contained program; driven by settings.json.
python 00_stage0_reproduction.py all                -> checks, smoke test, full run, figures
python 00_stage0_reproduction.py worker W NW [smoke]
============================================================================
"""

import sys, os, json, time, math, glob, itertools, traceback, subprocess, shutil
import numpy as np
from scipy.stats import qmc
from matplotlib.path import Path

SETTINGS = dict(
    # ---- model / training settings (fixed for the whole study) -------------
    # seed pool: 1234 = seed of the reference implementation (Stage 0 of Ex. 4.1);
    # the others come from numpy.random.SeedSequence(20260923).generate_state(9),
    # generated once and written here literally so they can never change.
    seeds=[
        1234,
        3154326823,
        3750868552,
        243740839,
        3730658245,
        3471616824,
        1792976967,
        2616433539,
        2214267315,
        3377612822,
    ],
    n_seeds=5,  # reproduction phase: 1234 + first 4 of the pool
    eps0=0.5,  # initial RBF width ~ 1.75 x mean center spacing (0.28-0.29)
    residual_form="scaled",  # PDE residual divided by a0 = Eq. (5) of the Part I report
    adam_iters=5000,  # 5000 Adam (lr 1e-3) then L-BFGS, as in Stages 0-2 of Ex. 4.1
    lbfgs_iters=20000,  # paper: L-BFGS, lr = 1, max 20 000 iterations
    dtype="float32",
    # ---- run control ----------------------------------------------------------
    hours=10.5,  # stop starting new runs after this (Kaggle limit is 12 h)
    prev="auto",  # "auto" = find previous results under PREV_RESULTS_DIR automatically
    # (upload example42_results.zip as a Dataset and add it as input)
    run_full=True,  # False = only checks + smoke test + Fig. 6
    ablation=True,  # also run the protocol ablation (paper-literal settings)
)
S = dict(SETTINGS)
if os.path.exists("settings.json"):  # optional overrides
    S.update(json.load(open("settings.json")))
# results of earlier stages are searched for under this directory
PREV_RESULTS_DIR = os.environ.get("PREV_RESULTS_DIR", "/kaggle/input")

ALPHAS, NS = [0.9, 0.5, 0.1], [10, 20, 40, 80]
TABLE1 = {
    "butterfly": dict(centers=140, n_int=150, n_bnd=50, n_test_int=230, n_test_bnd=100),
    "pentagram": dict(centers=140, n_int=150, n_bnd=50, n_test_int=220, n_test_bnd=100),
}
RES = "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ geometry
# Reconstructed from the paper's own figures (no formula is given in the paper):
# butterfly: Fig. 6a digitized + polar cosine fit (RMS 0.006 ~ 1.5 px)
# pentagram: Fig. 10, regular star, outer radius 1, inner radius 0.5
def butterfly_rho(th):
    return 0.764 - 0.109 * np.cos(th) + 0.253 * np.cos(2 * th) - 0.334 * np.cos(4 * th)


class Domain2D:
    def __init__(self, name, pts):
        self.name, self.poly = name, np.asarray(pts, float)
        closed = np.vstack([self.poly, self.poly[:1]])
        self.path = Path(closed)
        seg = np.diff(closed, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s_cum = np.r_[0.0, np.cumsum(self.seg_len)]
        self.perimeter = self.s_cum[-1]
        self.bbox = np.array([self.poly.min(0), self.poly.max(0)])

    def inside(self, xy, margin=0.0):
        xy = np.atleast_2d(xy)
        ins = self.path.contains_points(xy)
        if margin > 0:
            ins &= self.dist(xy) > margin
        return ins

    def dist(self, xy, chunk=256):
        if len(xy) > chunk:
            return np.concatenate([self.dist(xy[i : i + chunk]) for i in range(0, len(xy), chunk)])
        a = self.poly
        ab = np.roll(a, -1, 0) - a
        ap = xy[:, None] - a[None]
        t = np.clip((ap * ab[None]).sum(-1) / (ab**2).sum(-1)[None], 0, 1)
        return np.min(np.linalg.norm(xy[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1), 1)

    def boundary_at(self, s01):
        s = (np.asarray(s01) % 1.0) * self.perimeter
        i = np.clip(np.searchsorted(self.s_cum, s, side="right") - 1, 0, len(self.poly) - 1)
        loc = (s - self.s_cum[i]) / self.seg_len[i]
        a, b = self.poly[i], self.poly[(i + 1) % len(self.poly)]
        return a + loc[:, None] * (b - a)


def make_domain(name):
    if name == "butterfly":
        th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        r = butterfly_rho(th)
        return Domain2D(name, np.c_[r * np.cos(th), r * np.sin(th)])
    k = np.arange(10)
    ang = np.pi / 2 + k * np.pi / 5
    r = np.where(k % 2 == 0, 1.0, 0.5)
    return Domain2D(name, np.c_[r * np.cos(ang), r * np.sin(ang)])


# ------------------------------------------------------------------ sampling
MARGIN = 1e-3


def _radinv(i):
    f, r = 1.0, 0.0
    while i > 0:
        f /= 2
        r += f * (i % 2)
        i //= 2
    return r


def interior_points(dom, n, sampler, seed):
    lo, hi = dom.bbox
    m = n
    while True:
        if sampler == "lhs":
            u = qmc.LatinHypercube(d=2, seed=seed).random(m)
        else:
            u = np.c_[np.arange(m) / m, [_radinv(i) for i in range(m)]]
        xy = lo + u * (hi - lo)
        xy = xy[dom.inside(xy, MARGIN)]
        if len(xy) >= n:
            break
        m = int(m * 1.15) + 1
    if len(xy) == n:
        return xy
    if sampler == "hammersley":
        return xy[np.round(np.linspace(0, len(xy) - 1, n)).astype(int)]
    return xy[np.sort(np.random.default_rng(seed + 7).choice(len(xy), n, replace=False))]


def boundary_points(dom, n, sampler, seed):
    s = qmc.LatinHypercube(d=1, seed=seed).random(n)[:, 0] if sampler == "lhs" else np.arange(n) / n
    return dom.boundary_at(np.sort(s))


def uniform_test_points(dom, n_int, n_bnd):
    lo, hi = dom.bbox
    best = None
    for h in np.arange(0.25, 0.04, -0.0005):
        for off in (0.5, 0.25, 0.0):
            XX, YY = np.meshgrid(
                np.arange(lo[0] + off * h, hi[0], h), np.arange(lo[1] + off * h, hi[1], h)
            )
            xy = np.c_[XX.ravel(), YY.ravel()]
            xy = xy[dom.inside(xy)]
            if best is None or abs(len(xy) - n_int) < abs(len(best) - n_int):
                best = xy
            if len(best) == n_int or len(xy) > 1.5 * n_int:
                break
        if len(best) == n_int or len(xy) > 1.5 * n_int:
            break
    return best[dom.inside(best, MARGIN)], dom.boundary_at(np.arange(n_bnd) / n_bnd)


def point_sets(dom, sampler, seed):
    c = TABLE1[dom.name]
    ti, tb = uniform_test_points(dom, c["n_test_int"], c["n_test_bnd"])
    return dict(
        x_int=interior_points(dom, c["n_int"], sampler, seed),
        x_bnd=boundary_points(dom, c["n_bnd"], sampler, seed),
        xt_int=ti,
        xt_bnd=tb,
    )


def plotting_grid(dom, n=301):
    lo, hi = dom.bbox
    pad = 0.02 * (hi - lo)
    XX, YY = np.meshgrid(
        np.linspace(lo[0] - pad[0], hi[0] + pad[0], n),
        np.linspace(lo[1] - pad[1], hi[1] + pad[1], n),
    )
    return XX, YY, dom.inside(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)


# ------------------------------------------------------------------ problem
# u = (1 + t^(2-a)/G(3-a)) sin(k pi x) sin(k pi y); k = 1 is the paper exactly
def time_factor(t, a):
    return 1.0 + t ** (2 - a) / math.gamma(3 - a)


def exact_np(x, y, t, a, k=1):
    return time_factor(t, a) * np.sin(k * np.pi * x) * np.sin(k * np.pi * y)


def source_np(x, y, t, a, k=1):
    g = (
        t ** (1 - a) / math.gamma(2 - a)
        + t ** (2 - 2 * a) / math.gamma(3 - 2 * a)
        + 2 * k * k * np.pi**2 * time_factor(t, a)
    )
    return g * np.sin(k * np.pi * x) * np.sin(k * np.pi * y)


def l1_matrix(a, N, T=1.0):
    tau = T / N
    l = np.arange(N)
    c = tau ** (-a) / math.gamma(2 - a) * ((l + 1) ** (1 - a) - l ** (1 - a))
    W = np.zeros((N, N + 1))
    for n in range(1, N + 1):
        W[n - 1, n] = c[0]
        W[n - 1, 0] = -c[n - 1]
        for k in range(1, n):
            W[n - 1, k] = -(c[n - k - 1] - c[n - k])
    return W, c[0]


# ------------------------------------------------------------------ paper values
_T3 = {
    "gaussian": {
        0.9: [
            (1.1896e-03, 1.2430e-03, 9.9698e-04, 1.0759e-03),
            (7.9920e-04, 7.8644e-04, 7.1933e-04, 7.6487e-04),
            (7.7247e-04, 8.3923e-04, 5.8280e-04, 6.3336e-04),
            (6.2803e-04, 7.2099e-04, 6.6671e-04, 7.5954e-04),
        ],
        0.5: [
            (2.5386e-03, 2.5734e-03, 2.5234e-03, 2.6213e-03),
            (2.2746e-03, 2.3262e-03, 1.8144e-03, 1.9794e-03),
            (1.7236e-03, 1.7859e-03, 2.3600e-03, 2.5312e-03),
            (1.6683e-03, 1.7387e-03, 2.1810e-03, 2.3803e-03),
        ],
        0.1: [
            (7.8079e-03, 7.8360e-03, 7.3388e-03, 7.2927e-03),
            (7.2289e-03, 7.4549e-03, 7.1739e-03, 7.6650e-03),
            (7.2360e-03, 7.7550e-03, 6.6432e-03, 7.3525e-03),
            (9.7012e-03, 1.0492e-02, 9.3428e-03, 1.0467e-02),
        ],
    },
    "iq": {
        0.9: [
            (2.9474e-03, 3.2053e-03, 2.6650e-03, 3.0881e-03),
            (1.9390e-03, 1.9548e-03, 2.0647e-03, 2.4954e-03),
            (2.1242e-03, 2.3363e-03, 1.4843e-03, 1.8241e-03),
            (1.1232e-03, 1.3141e-03, 9.6941e-04, 1.1898e-03),
        ],
        0.5: [
            (3.4562e-03, 3.6386e-03, 3.9978e-03, 4.5384e-03),
            (3.0718e-03, 3.1660e-03, 3.1600e-03, 3.6368e-03),
            (3.6941e-03, 3.8564e-03, 3.5613e-03, 4.1523e-03),
            (2.9099e-03, 3.1946e-03, 2.5779e-03, 2.9616e-03),
        ],
        0.1: [
            (1.2829e-02, 1.3624e-02, 1.1685e-02, 1.2942e-02),
            (1.1913e-02, 1.2848e-02, 1.1103e-02, 1.2596e-02),
            (1.3710e-02, 1.4546e-02, 1.1815e-02, 1.3261e-02),
            (1.4894e-02, 1.5738e-02, 1.5877e-02, 1.7670e-02),
        ],
    },
    "imq": {
        0.9: [
            (7.6892e-03, 8.1049e-03, 5.2919e-03, 6.4335e-03),
            (7.2354e-03, 7.8156e-03, 4.6023e-03, 5.5588e-03),
            (4.9545e-03, 5.3261e-03, 3.8225e-03, 4.5767e-03),
            (3.1285e-03, 3.5456e-03, 2.4520e-03, 3.0763e-03),
        ],
        0.5: [
            (1.0266e-02, 1.0746e-02, 9.9753e-03, 1.1575e-02),
            (7.6002e-03, 7.9072e-03, 6.6604e-03, 7.6575e-03),
            (8.3840e-03, 8.9802e-03, 7.0382e-03, 8.2189e-03),
            (7.3668e-03, 7.9242e-03, 7.1735e-03, 8.4650e-03),
        ],
        0.1: [
            (1.8512e-02, 1.9418e-02, 1.9153e-02, 2.1624e-02),
            (2.0681e-02, 2.2114e-02, 2.3820e-02, 2.6940e-02),
            (2.9446e-02, 3.0767e-02, 2.3077e-02, 2.5745e-02),
            (2.6127e-02, 2.7996e-02, 2.6689e-02, 3.0265e-02),
        ],
    },
}
_T4 = {
    0.9: [
        (9.2551e-04, 1.0415e-03),
        (8.9037e-04, 8.8786e-04),
        (7.1663e-04, 7.4893e-04),
        (6.1986e-04, 6.3985e-04),
    ],
    0.5: [
        (2.6012e-03, 2.6935e-03),
        (2.2370e-03, 2.2557e-03),
        (1.5936e-03, 1.6594e-03),
        (1.5980e-03, 1.6502e-03),
    ],
    0.1: [
        (5.8492e-03, 5.8968e-03),
        (5.6118e-03, 5.6334e-03),
        (8.1776e-03, 8.1412e-03),
        (9.5921e-03, 9.7807e-03),
    ],
}
TABLE3 = {k: {a: dict(zip(NS, v)) for a, v in d.items()} for k, d in _T3.items()}
TABLE4 = {a: dict(zip(NS, v)) for a, v in _T4.items()}


# ------------------------------------------------------------------ jobs
def all_jobs():
    base = dict(
        T=1.0,
        k=1,
        dtype=S["dtype"],
        residual_form=S["residual_form"],
        adam_iters=S["adam_iters"],
        adam_lr=1e-3,
        lbfgs_iters=S["lbfgs_iters"],
        lbfgs_lr=1.0,
        tol_grad=1e-9,
        tol_change=1e-12,
        eps0=S["eps0"],
        arch="rbf",
        centers=140,
    )
    jobs = []
    for s in S["seeds"][: S["n_seeds"]]:  # seed-major: first seed complete first
        for kern, samp, a, N in itertools.product(
            ["gaussian", "iq", "imq"], ["lhs", "hammersley"], ALPHAS, NS
        ):
            jobs.append(
                dict(base, domain="butterfly", kernel=kern, sampler=samp, alpha=a, N=N, seed=s)
            )
        for a, N in itertools.product(ALPHAS, NS):
            jobs.append(
                dict(
                    base, domain="pentagram", kernel="gaussian", sampler="lhs", alpha=a, N=N, seed=s
                )
            )
    if S.get("ablation"):
        # Protocol ablation (runs AFTER the reproduction jobs): which of our two
        # deviations from the literal paper text explains the gap to the paper?
        #   residual_form: scaled (ours) vs paper ;  adam_iters: 5000 (ours) vs 0 (paper text)
        for s in S["seeds"][:3]:
            for rf, ad in [("paper", 0), ("scaled", 0), ("paper", 5000)]:
                for a, N in itertools.product(ALPHAS, [10, 40, 80]):
                    jobs.append(
                        dict(
                            base,
                            residual_form=rf,
                            adam_iters=ad,
                            domain="butterfly",
                            kernel="gaussian",
                            sampler="lhs",
                            alpha=a,
                            N=N,
                            seed=s,
                        )
                    )
    return jobs


def job_id(j):
    return (
        f"{j['domain']}_{j['kernel']}_{j['sampler']}_a{j['alpha']}_N{j['N']}_k{j['k']}"
        f"_{j['residual_form']}_ad{j['adam_iters']}_{j['dtype']}_s{j['seed']}"
    )


def read_runs(folder):
    rows = []
    for fn in glob.glob(os.path.join(folder, "runs_w*.jsonl")):
        for line in open(fn):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


# ------------------------------------------------------------------ one training run
def train_one(cfg, dom, pts, device, grid):
    import torch
    import torch.nn as nn

    dtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    a, N, T, k = cfg["alpha"], cfg["N"], cfg["T"], cfg["k"]
    tgrid = np.linspace(0, T, N + 1)
    P = len(pts["x_int"])
    tt = lambda arr: torch.tensor(arr, dtype=dtype, device=device)
    st = lambda xy, t: np.hstack(
        [np.repeat(xy[None], len(t), 0).reshape(-1, 2), np.repeat(t, len(xy))[:, None]]
    )

    def ex_t(Z):  # exact solution, torch (differentiable)
        return (
            (1.0 + Z[:, 2] ** (2 - a) / math.gamma(3 - a))
            * torch.sin(k * math.pi * Z[:, 0])
            * torch.sin(k * math.pi * Z[:, 1])
        )

    Zi_np = st(pts["x_int"], tgrid)
    Zb_np = st(pts["x_bnd"], tgrid[1:])
    Z_int = tt(Zi_np).requires_grad_(True)
    Z_bnd = tt(Zb_np)
    W_np, a0 = l1_matrix(a, N, T)
    W = tt(W_np)
    f_int = tt(source_np(Zi_np[:, 0], Zi_np[:, 1], Zi_np[:, 2], a, k).reshape(N + 1, P)[1:])
    u0 = tt(exact_np(Zi_np[:P, 0], Zi_np[:P, 1], 0.0, a, k))
    g_bnd = tt(exact_np(Zb_np[:, 0], Zb_np[:, 1], Zb_np[:, 2], a, k))
    scale = 1.0 / a0 if cfg["residual_form"] == "scaled" else 1.0

    class RBF(nn.Module):
        def __init__(s):
            super().__init__()
            lo = torch.tensor([*dom.bbox[0], 0.0])
            hi = torch.tensor([*dom.bbox[1], T])
            c = (lo + hi) / 2 + (hi - lo) / 4 * torch.randn(cfg["centers"], 3, generator=gen)
            s.c = nn.Parameter(torch.max(torch.min(c, hi), lo))
            s.le = nn.Parameter(torch.full((cfg["centers"],), math.log(cfg["eps0"])))
            s.w = nn.Parameter(
                torch.randn(cfg["centers"], 1, generator=gen) / math.sqrt(cfg["centers"])
            )

        def forward(s, z):
            q = ((z[:, None, :] - s.c[None]) ** 2).sum(-1) * torch.exp(-2 * s.le)[None]
            h = (
                torch.exp(-q)
                if cfg["kernel"] == "gaussian"
                else (1 / (1 + q) if cfg["kernel"] == "iq" else torch.rsqrt(1 + q))
            )
            return h @ s.w

    model = RBF().to(device=device, dtype=dtype)

    def losses(net):
        u = net(Z_int).reshape(-1)
        g = torch.autograd.grad(u.sum(), Z_int, create_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0].sum(), Z_int, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(g[:, 1].sum(), Z_int, create_graph=True)[0][:, 1]
        U = u.reshape(N + 1, P)
        r = (
            g[:, 2].reshape(N + 1, P)[1:] + W @ U - (uxx + uyy).reshape(N + 1, P)[1:] - f_int
        ) * scale
        return (
            ((U[0] - u0) ** 2).mean(),
            ((net(Z_bnd).reshape(-1) - g_bnd) ** 2).mean(),
            (r**2).mean(),
        )

    ex = [v.item() for v in losses(lambda Z: ex_t(Z)[:, None])]  # correctness guard
    hist, it = [], [0]

    def total():
        li, lb, lf = losses(model)
        if it[0] % 100 == 0:
            hist.append((it[0], li.item(), lb.item(), lf.item()))
        return li + lb + lf

    t0 = time.time()
    if cfg["adam_iters"] > 0:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"])
        for _ in range(cfg["adam_iters"]):
            opt.zero_grad()
            L = total()
            L.backward()
            opt.step()
            it[0] += 1
    t_adam = time.time() - t0
    me = int(cfg["lbfgs_iters"] * 1.25)
    nev = [0]
    lb = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg["lbfgs_lr"],
        max_iter=cfg["lbfgs_iters"],
        max_eval=me,
        history_size=50,
        tolerance_grad=cfg["tol_grad"],
        tolerance_change=cfg["tol_change"],
        line_search_fn="strong_wolfe",
    )

    def closure():
        lb.zero_grad()
        L = total()
        L.backward()
        nev[0] += 1
        it[0] += 1
        return L

    lb.step(closure)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    n_iter = int(lb.state[lb._params[0]].get("n_iter", 0))
    gmax = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    reason = (
        "max_iter"
        if n_iter >= cfg["lbfgs_iters"]
        else (
            "max_eval"
            if nev[0] >= me
            else "grad_tol" if gmax <= cfg["tol_grad"] else "change_tol_or_stall"
        )
    )
    li, lbc, lf = [v.item() for v in losses(model)]

    def predict(Z):
        with torch.no_grad():
            return np.concatenate(
                [
                    model(tt(Z[i : i + 20000])).reshape(-1).cpu().numpy()
                    for i in range(0, len(Z), 20000)
                ]
            ).astype(np.float64)

    def rel(Z):
        u = exact_np(Z[:, 0], Z[:, 1], Z[:, 2], a, k)
        return float(np.linalg.norm(predict(Z) - u) / np.linalg.norm(u))

    out = dict(
        E_train=rel(st(np.vstack([pts["x_int"], pts["x_bnd"]]), tgrid)),
        E_test=rel(st(np.vstack([pts["xt_int"], pts["xt_bnd"]]), np.linspace(0, T, 100))),
        L_ic=li,
        L_bc=lbc,
        L_f=lf,
        wall_time=wall,
        adam_time=t_adam,
        lbfgs_n_iter=n_iter,
        lbfgs_evals=nev[0],
        lbfgs_stop=reason,
        grad_max=gmax,
        n_params=sum(p.numel() for p in model.parameters()),
        exact_L_ic=ex[0],
        exact_L_bc=ex[1],
        exact_L_f=ex[2],
    )
    XX, YY, mask = grid
    Zp = np.c_[XX[mask], YY[mask], np.full(mask.sum(), T)]
    err = np.abs(predict(Zp) - exact_np(Zp[:, 0], Zp[:, 1], Zp[:, 2], a, k)).astype(np.float32)
    return out, dict(loss_hist=np.array(hist, np.float32), abs_err_final=err)


# ------------------------------------------------------------------ worker
def worker(w, nw, smoke=False, hours=10.5):
    import torch

    out_dir = "results_smoke" if smoke else RES
    os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
    jobs = all_jobs()
    if smoke:
        jobs = [dict(j, adam_iters=50, lbfgs_iters=50) for j in jobs if j["N"] in (10, 80)][:2]
    done = {r["id"] for r in read_runs(out_dir)}
    mine = [j for i, j in enumerate(jobs) if i % nw == w and job_id(j) not in done]
    ng = torch.cuda.device_count()
    device = torch.device(f"cuda:{w % ng}" if ng else "cpu")
    log(f"[w{w}] device={device} jobs to do: {len(mine)}")
    doms, grids, t_start = {}, {}, time.time()
    for j in mine:
        if (time.time() - t_start) / 3600 > hours:
            log(f"[w{w}] time budget reached")
            break
        if j["domain"] not in doms:
            doms[j["domain"]] = make_domain(j["domain"])
            grids[j["domain"]] = plotting_grid(doms[j["domain"]])
        jid = job_id(j)
        try:
            res, extras = train_one(
                j,
                doms[j["domain"]],
                point_sets(doms[j["domain"]], j["sampler"], j["seed"]),
                device,
                grids[j["domain"]],
            )
        except Exception as e:
            traceback.print_exc()
            res, extras = dict(error=repr(e)), {}
        np.savez_compressed(os.path.join(out_dir, "fields", jid + ".npz"), **extras)
        with open(os.path.join(out_dir, f"runs_w{w}.jsonl"), "a") as f:
            f.write(json.dumps({"id": jid, **j, **res}) + "\n")
        log(
            f"[w{w}] {jid} E={res.get('E_train', np.nan):.3e} Ehat={res.get('E_test', np.nan):.3e} "
            f"{res.get('wall_time', 0):.0f}s {res.get('lbfgs_stop', res.get('error', ''))}"
        )


# ------------------------------------------------------------------ figures & tables
def figures(out="figures"):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    KN = {"gaussian": "Gaussian", "iq": "Inverse quadratic", "imq": "Inverse multiquadric"}
    SN = {"lhs": "LHS points", "hammersley": "Hammersley points"}
    seed0 = S["seeds"][0]
    # Fig. 6
    dom = make_domain("butterfly")
    poly = np.vstack([dom.poly, dom.poly[:1]])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, samp, title in zip(
        axs,
        ["lhs", "hammersley", None],
        ["(a) LHS points", "(b) Hammersley points", "(c) Uniform test data"],
    ):
        p = point_sets(dom, samp or "lhs", seed0)
        xi, xb = (p["x_int"], p["x_bnd"]) if samp else (p["xt_int"], p["xt_bnd"])
        ax.fill(poly[:, 0], poly[:, 1], color="lightcyan", alpha=0.6)
        ax.plot(poly[:, 0], poly[:, 1], label="Butterfly Boundary")
        ax.scatter(xb[:, 0], xb[:, 1], s=14, c="blue", zorder=3, label="Boundary Points")
        ax.scatter(xi[:, 0], xi[:, 1], s=10, c="red", zorder=3, label="Interior Points")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{title}  [{len(xi)} / {len(xb)}]")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(f"{out}/fig6_points.png", dpi=200)
    plt.close(fig)

    rows = [r for r in read_runs(RES) if "error" not in r]
    if not rows:
        log("no results yet -> only Fig. 6")
        return
    df_all = pd.DataFrame(rows)
    main = (df_all.residual_form == S["residual_form"]) & (df_all.adam_iters == S["adam_iters"])
    df = df_all[main]
    abl = df_all[
        (df_all.domain == "butterfly")
        & (df_all.kernel == "gaussian")
        & (df_all.sampler == "lhs")
        & df_all.N.isin([10, 40, 80])
    ]
    if len(abl) and (~main).any():
        seeds3 = S["seeds"][:3]
        abl = abl[abl.seed.isin(seeds3)]
        g = (
            abl.groupby(["residual_form", "adam_iters", "alpha", "N"])
            .E_test.agg(["mean", "std", "size"])
            .reset_index()
        )
        g["paper"] = [TABLE3["gaussian"][a][N][1] for a, N in zip(g.alpha, g.N)]
        g["ratio_ours_over_paper"] = g["mean"] / g["paper"]
        g["protocol"] = g.residual_form + " / adam=" + g.adam_iters.astype(str)
        g.to_csv(f"{out}/ablation.csv", index=False)
        piv = g.pivot_table(
            index=["alpha", "N"], columns="protocol", values="ratio_ours_over_paper"
        )
        piv.to_csv(f"{out}/ablation_ratio_pivot.csv")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
        for ax, a in zip(axs, ALPHAS):
            for prot, gg in g[g.alpha == a].groupby("protocol"):
                ax.errorbar(
                    gg.N, gg["mean"], yerr=gg["std"].fillna(0), marker="o", capsize=3, label=prot
                )
            pv = [TABLE3["gaussian"][a][N][1] for N in [10, 40, 80]]
            ax.plot([10, 40, 80], pv, "k--", marker="s", label="paper (single run)")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"alpha = {a}")
            ax.set_xlabel("N")
        axs[0].set_ylabel("test error (Gaussian, LHS, butterfly)")
        axs[0].legend(fontsize=8)
        fig.suptitle("Protocol ablation: residual scaling x Adam warm-up vs published values")
        fig.tight_layout()
        fig.savefig(f"{out}/ablation.png", dpi=200)
        plt.close(fig)
        log("ABLATION (ours/paper, rows alpha,N):\n" + piv.round(2).to_string())

    def field(rid):
        fn = f"{RES}/fields/{rid}.npz"
        return np.load(fn)["abs_err_final"] if os.path.exists(fn) else None

    def pick(sub):
        s = sub[sub.seed == seed0]
        return s.iloc[0] if len(s) else sub.sort_values("E_test").iloc[len(sub) // 2]

    def emap(ax, dom, err, title):
        XX, YY, mask = plotting_grid(dom)
        Z = np.full(XX.shape, np.nan)
        Z[mask] = err
        pc = ax.pcolormesh(XX, YY, Z, cmap="jet", shading="auto", vmin=0)
        pp = np.vstack([dom.poly, dom.poly[:1]])
        ax.plot(pp[:, 0], pp[:, 1], "k", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    # Figs. 7-9 (N = 80)
    b = df[(df.domain == "butterfly") & (df.N == 80)]
    for a in ALPHAS:
        fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
        for i, samp in enumerate(["lhs", "hammersley"]):
            for jx, kern in enumerate(["gaussian", "iq", "imq"]):
                ax = axs[i, jx]
                sub = b[(b.alpha == a) & (b.sampler == samp) & (b.kernel == kern)]
                err = field(pick(sub)["id"]) if len(sub) else None
                if err is None:
                    ax.axis("off")
                    continue
                emap(ax, dom, err, "Error at final time")
                if i == 0:
                    ax.annotate(
                        KN[kern], (0.5, 1.18), xycoords="axes fraction", ha="center", fontsize=15
                    )
                if jx == 0:
                    ax.annotate(
                        SN[samp],
                        (-0.35, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        rotation=90,
                        fontsize=15,
                    )
        fig.suptitle(f"Point-wise error at final time, alpha = {a}, N = 80")
        fig.tight_layout()
        fig.savefig(f"{out}/fig7-9_alpha{a}.png", dpi=200)
        plt.close(fig)
    # Fig. 10
    pdom = make_domain("pentagram")
    p = df[(df.domain == "pentagram") & (df.N == 80)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, a in zip(axs, ALPHAS):
        sub = p[p.alpha == a]
        err = field(pick(sub)["id"]) if len(sub) else None
        if err is None:
            ax.axis("off")
            continue
        emap(ax, pdom, err, "Absolute error at final time")
        ax.set_xlabel(f"alpha = {a}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out}/fig10_pentagram.png", dpi=200)
    plt.close(fig)

    # Tables 3, 4 (mean (std) over seeds) + comparison with published values
    fmt = lambda v: (
        "--"
        if len(v) == 0
        else (f"{v.mean():.4e}" + (f" ({v.std(ddof=1):.1e})" if len(v) > 1 else ""))
    )
    t3, chk = [], []
    b = df[df.domain == "butterfly"]
    for kern in ["gaussian", "iq", "imq"]:
        for a in ALPHAS:
            for N in NS:
                row = dict(kernel=kern, alpha=a, N=N)
                for samp, (iE, iH) in [("lhs", (0, 1)), ("hammersley", (2, 3))]:
                    sub = b[(b.kernel == kern) & (b.alpha == a) & (b.N == N) & (b.sampler == samp)]
                    row[f"E_{samp}"], row[f"Ehat_{samp}"] = fmt(sub.E_train), fmt(sub.E_test)
                    row[f"paper_Ehat_{samp}"] = TABLE3[kern][a][N][iH]
                    if len(sub):
                        chk.append(
                            (
                                kern,
                                a,
                                N,
                                samp,
                                TABLE3[kern][a][N][iH],
                                sub.E_test.mean(),
                                sub.E_test.min(),
                                sub.E_test.max(),
                                len(sub),
                            )
                        )
                t3.append(row)
    pd.DataFrame(t3).to_csv(f"{out}/table3.csv", index=False)
    t4 = []
    p = df[df.domain == "pentagram"]
    for N in NS:
        row = dict(N=N)
        for a in ALPHAS:
            sub = p[(p.alpha == a) & (p.N == N)]
            row[f"E_a{a}"], row[f"Ehat_a{a}"], row[f"paper_Ehat_a{a}"] = (
                fmt(sub.E_train),
                fmt(sub.E_test),
                TABLE4[a][N][1],
            )
            if len(sub):
                chk.append(
                    (
                        "pentagram",
                        a,
                        N,
                        "lhs",
                        TABLE4[a][N][1],
                        sub.E_test.mean(),
                        sub.E_test.min(),
                        sub.E_test.max(),
                        len(sub),
                    )
                )
        t4.append(row)
    pd.DataFrame(t4).to_csv(f"{out}/table4.csv", index=False)
    d = pd.DataFrame(
        chk, columns=["kernel", "alpha", "N", "sampler", "paper", "mean", "min", "max", "n_seeds"]
    )
    d["ratio_ours_over_paper"] = d["mean"] / d["paper"]
    d["paper_inside_seed_range"] = (d.paper >= d["min"]) & (d.paper <= d["max"])
    d.to_csv(f"{out}/repro_check.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    col = {"gaussian": "tab:blue", "iq": "tab:red", "imq": "tab:purple", "pentagram": "tab:green"}
    for (kk, a), g in d.groupby(["kernel", "alpha"]):
        ax.errorbar(
            g.paper,
            g["mean"],
            yerr=[g["mean"] - g["min"], g["max"] - g["mean"]],
            fmt={0.9: "o", 0.5: "s", 0.1: "^"}[a],
            color=col[kk],
            ms=5,
            capsize=2,
            lw=0.8,
            label=f"{kk}, a={a}",
        )
    lim = [min(d.paper.min(), d["min"].min()) / 1.5, max(d.paper.max(), d["max"].max()) * 1.5]
    ax.plot(lim, lim, "k-", lw=1)
    ax.fill_between(lim, [l / 2 for l in lim], [l * 2 for l in lim], color="gray", alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlabel("published test error")
    ax.set_ylabel("reproduced test error (mean, min-max)")
    ax.set_title("Reproduction check (band = factor 2)")
    fig.tight_layout()
    fig.savefig(f"{out}/repro_check.png", dpi=200)
    plt.close(fig)
    df.groupby(["domain", "kernel", "sampler", "alpha", "N"]).agg(
        n=("seed", "size"),
        wall_mean=("wall_time", "mean"),
        lbfgs_n_iter=("lbfgs_n_iter", "mean"),
        stall_frac=("lbfgs_stop", lambda s: (s == "change_tol_or_stall").mean()),
    ).reset_index().to_csv(f"{out}/diagnostics.csv", index=False)
    log(
        f"REPRODUCTION: {len(d)} cells | median ours/paper = {d.ratio_ours_over_paper.median():.2f} | "
        f"within factor 2: {(np.abs(np.log2(d.ratio_ours_over_paper)) <= 1).mean():.0%} | "
        f"paper inside seed range: {d.paper_inside_seed_range.mean():.0%}"
    )


# ------------------------------------------------------------------ checks & orchestration
def offline_checks():
    from scipy.integrate import quad

    for a in ALPHAS:
        t, x, y = 0.7, 0.3, -0.4
        gp = lambda s: s ** (1 - a) / math.gamma(2 - a)
        cap = quad(lambda s: (t - s) ** (-a) * gp(s), 0, t, limit=200)[0] / math.gamma(1 - a)
        Sx = np.sin(np.pi * x) * np.sin(np.pi * y)
        res = gp(t) * Sx + cap * Sx + 2 * np.pi**2 * time_factor(t, a) * Sx - source_np(x, y, t, a)
        assert abs(res) < 1e-6, f"source term check failed (alpha={a}): {res}"
        for N in NS:
            assert np.allclose(l1_matrix(a, N)[0].sum(1), 0), "L1 rows must sum to zero"
    for name in ["butterfly", "pentagram"]:
        p = point_sets(make_domain(name), "lhs", S["seeds"][0])
        log(name, {k: len(v) for k, v in p.items()})
        assert len(p["x_int"]) == 150 and len(p["x_bnd"]) == 50
    log("OFFLINE CHECKS PASSED (source term, Caputo derivative, L1 matrix, point counts)")


def main_all():
    import torch

    prev = S["prev"]
    if prev == "auto":
        hits = sorted(
            glob.glob(os.path.join(PREV_RESULTS_DIR, "**", "runs_w*.jsonl"), recursive=True)
        )
        prev = os.path.dirname(hits[0]) if hits else ""
        log("auto-detected previous results:", prev or "none (fresh start)")
    if prev and os.path.isdir(prev):
        shutil.copytree(prev, RES, dirs_exist_ok=True)
        log("restored", len(read_runs(RES)), "finished runs from", prev)
    log(
        "GPUs:",
        torch.cuda.device_count(),
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )
    offline_checks()
    # smoke test: 2 short runs; stop everything if they fail
    shutil.rmtree("results_smoke", ignore_errors=True)
    subprocess.run([sys.executable, __file__, "worker", "0", "1", "smoke"], check=False)
    sm = read_runs("results_smoke")
    bad = [
        r["id"] + ": " + r.get("error", "exact-solution check failed")
        for r in sm
        if "error" in r
        or not (r["exact_L_ic"] < 1e-10 and r["exact_L_bc"] < 1e-10 and np.isfinite(r["E_train"]))
    ]
    if len(sm) != 2 or bad:
        raise SystemExit(f"SMOKE TEST FAILED -> full run NOT started: {bad or 'runs missing'}")
    log("SMOKE TEST PASSED")
    if S["run_full"]:
        nw = max(1, torch.cuda.device_count())
        t0 = time.time()
        procs = [
            subprocess.Popen([sys.executable, __file__, "worker", str(w), str(nw), "full"])
            for w in range(nw)
        ]
        total = len(all_jobs())
        while any(p.poll() is None for p in procs):
            time.sleep(600)
            log(f"progress: {len(read_runs(RES))}/{total} runs | {(time.time() - t0)/3600:.1f} h")
            if time.time() - t0 > (S["hours"] + 0.7) * 3600:
                for p in procs:
                    p.terminate()
                log("hard stop")
                break
    figures()
    shutil.make_archive("example42_results", "zip", ".", RES)
    log("DONE. results/, figures/, example42_results.zip are in the working directory")


if __name__ == "__main__":
    if sys.argv[1] == "all":
        main_all()
    else:
        worker(int(sys.argv[2]), int(sys.argv[3]), smoke=(sys.argv[4] == "smoke"), hours=S["hours"])
