"""Example 4.2 (2D) - Stage 2: budget-matched comparison (equal parameters / equal training time).

============================================================================
Example 4.2 -- STAGE 2: budget-matched comparison of RBF-fPINN and MLP-fPINN
Same protocol as Stage 2 of Example 4.1:
Scenario A (parameter-matched): RBF retrained with the number of centers that
matches each of the 3 MLP configurations selected in Stage 1 (5 params/center);
MLP results are taken from Stage 1 (same seeds, same points); eps0 fixed (0.5).
Scenario B (time-matched): for each (alpha, seed) the RBF baseline (140 centers)
is retrained and timed; the Stage-1 winning MLP is restarted with fresh seeds
until its cumulative time reaches the RBF time (checked between restarts,
max 8 restarts); the restart with the lowest final TRAINING loss is kept.
Single N = 80 (paper setting), alpha in {0.9, 0.5, 0.1}, butterfly, LHS;
Scenario A: 5 seeds (paired with Stage 1), Scenario B: 10 seeds (as in Example 4.1).
Every run also stores the time-resolved test error E(t) (100 times).
python 02_stage2_budget_matched.py all  |  python 02_stage2_budget_matched.py worker W NW PHASE HOURS
============================================================================
"""

import sys, os, json, time, math, glob, traceback, subprocess, shutil
import numpy as np
from scipy.stats import qmc
from matplotlib.path import Path

SETTINGS = dict(
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
    hours=10.5,  # stop starting new runs after this (Kaggle limit is 12 h)
    run=True,  # False = report only (no training)
)
S = dict(SETTINGS)
if os.path.exists("settings.json"):  # optional overrides
    S.update(json.load(open("settings.json")))
# results of earlier stages are searched for under this directory
PREV_RESULTS_DIR = os.environ.get("PREV_RESULTS_DIR", "/kaggle/input")

ALPHAS = [0.9, 0.5, 0.1]
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


# ------------------------------------------------------------------ validation set
# Fixed, seed-independent, disjoint from training and test points. Used ONLY for
# selecting MLP configurations; the test set is used only for final reporting.
_VAL = {}


def validation_set(dom):
    if dom.name not in _VAL:
        xi = interior_points(dom, 300, "lhs", 777001)
        xb = dom.boundary_at(np.sort(np.random.default_rng(777002).random(100)))
        tv = np.sort(np.random.default_rng(777003).uniform(0.0, 1.0, 20))
        _VAL[dom.name] = (np.vstack([xi, xb]), tv)
    return _VAL[dom.name]


def read_runs(folder):
    best = {}
    for fn in sorted(glob.glob(os.path.join(folder, "runs_w*.jsonl"))):
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] not in best or ("error" in best[r["id"]] and "error" not in r):
                best[r["id"]] = r
    return list(best.values())


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

    def ex_t(Z):
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

    class MLPNet(nn.Module):
        """tanh/sin MLP, Xavier-normal weights, zero biases."""

        def __init__(s):
            super().__init__()
            d_in = 3
            dims = [d_in] + [cfg["width"]] * cfg["depth"] + [1]
            s.L = nn.ModuleList(nn.Linear(p, q) for p, q in zip(dims[:-1], dims[1:]))
            with torch.no_grad():
                for L in s.L:
                    L.weight.copy_(
                        math.sqrt(2.0 / (L.in_features + L.out_features))
                        * torch.randn(L.weight.shape, generator=gen)
                    )
                    L.bias.zero_()
            s.act = torch.tanh if cfg["act"] == "tanh" else torch.sin

        def forward(s, z):
            for L in s.L[:-1]:
                z = s.act(L(z))
            return s.L[-1](z)

    model = (RBF() if cfg["arch"] == "rbf" else MLPNet()).to(device=device, dtype=dtype)

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

    ex = [v.item() for v in losses(lambda Z: ex_t(Z)[:, None])]
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

    xv, tv = validation_set(dom)
    out = dict(
        E_train=rel(st(np.vstack([pts["x_int"], pts["x_bnd"]]), tgrid)),
        E_val=rel(st(xv, tv)),
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
    # time-resolved relative L2 error on the test points (100 uniform times)
    t_eval = np.linspace(0, T, 100)
    xt = np.vstack([pts["xt_int"], pts["xt_bnd"]])
    Zt = st(xt, t_eval)
    ue = exact_np(Zt[:, 0], Zt[:, 1], Zt[:, 2], a, k).reshape(100, -1)
    up = predict(Zt).reshape(100, -1)
    err_t = (
        np.linalg.norm(up - ue, axis=1) / np.maximum(np.linalg.norm(ue, axis=1), 1e-12)
    ).astype(np.float32)
    out["L_total"] = li + lbc + lf
    XX, YY, mask = grid
    Zp = np.c_[XX[mask], YY[mask], np.full(mask.sum(), T)]
    err = np.abs(predict(Zp) - exact_np(Zp[:, 0], Zp[:, 1], Zp[:, 2], a, k)).astype(np.float32)
    return out, dict(
        loss_hist=np.array(hist, np.float32),
        abs_err_final=err,
        err_t=err_t,
        t_eval=t_eval.astype(np.float32),
    )


# ------------------------------------------------------------------ protocol
BASE = dict(
    T=1.0,
    k=1,
    dtype="float32",
    residual_form="scaled",
    adam_iters=5000,
    adam_lr=1e-3,
    lbfgs_iters=20000,
    lbfgs_lr=1.0,
    tol_grad=1e-9,
    tol_change=1e-12,
    eps0=0.5,
    domain="butterfly",
    sampler="lhs",
    N=80,
)
SEL = [
    dict(arch="mlp", depth=2, width=150, act="tanh"),  # Stage-1 selection (winner first)
    dict(arch="mlp", depth=1, width=100, act="tanh"),
    dict(arch="mlp", depth=2, width=100, act="tanh"),
]
MAX_RESTARTS = 8


def n_mlp_params(c):
    dims = [3] + [c["width"]] * c["depth"] + [1]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def matched_centers(c):  # RBF in (x, y, t): 3 center coords + 1 width + 1 weight
    return int(round(n_mlp_params(c) / 5))


def ckey(c):
    if c["arch"] == "mlp":
        return f"mlp{c['depth']}x{c['width']}_{c['act']}"
    return f"rbf_{c['kernel']}_C{c.get('centers', 140)}"


def job_id(j):
    if j.get("scenario") == "B":
        return f"S2B_{j['domain']}_a{j['alpha']}_N{j['N']}_s{j['seed']}"
    tag = ckey(j)
    if j["arch"] == "rbf" and j.get("centers", 140) == 140 and j.get("kernel") == "gaussian":
        tag = "gaussian"  # identical id scheme to the reproduction runs
    elif j["arch"] == "rbf":
        tag = f"gaussian_C{j['centers']}"
    return (
        f"{j['domain']}_{tag}_{j['sampler']}_a{j['alpha']}_N{j['N']}_k{j['k']}"
        f"_{j['residual_form']}_ad{j['adam_iters']}_{j['dtype']}_s{j['seed']}"
    )


def mlp_stage1_id(c, a, s):  # id of the matching Stage-1 MLP run
    return f"butterfly_{ckey(c)}_lhs_a{a}_N80_k1_scaled_ad5000_float32_s{s}"


def phase_jobs(phase):
    seeds = S["seeds"][:5]
    if phase == "B":  # 10 seeds, as in Scenario B of Example 4.1 (RBF is retrained here)
        return [dict(BASE, scenario="B", alpha=a, seed=s) for s in S["seeds"][:10] for a in ALPHAS]
    if phase == "A":  # cheapest budget first; within a budget seed-major
        jobs = []
        for c in sorted(SEL, key=n_mlp_params):
            for s in seeds:
                for a in ALPHAS:
                    jobs.append(
                        dict(
                            BASE,
                            scenario="A",
                            arch="rbf",
                            kernel="gaussian",
                            centers=matched_centers(c),
                            match=ckey(c),
                            alpha=a,
                            seed=s,
                        )
                    )
        return jobs
    raise ValueError(phase)


# ------------------------------------------------------------------ one Scenario-B job
def run_B(j, dom, device, grid):
    pts = point_sets(dom, "lhs", j["seed"])
    rcfg = dict(BASE, arch="rbf", kernel="gaussian", centers=140, alpha=j["alpha"], seed=j["seed"])
    r_out, r_ext = train_one(rcfg, dom, pts, device, grid)
    budget = r_out["wall_time"]
    restarts, spent, best, best_ext = [], 0.0, None, None
    for r in range(MAX_RESTARTS):
        if r > 0 and spent >= budget:
            break
        mseed = int(j["seed"]) + 1000 * r  # restart 0 uses the paired seed itself
        mcfg = dict(BASE, **SEL[0], alpha=j["alpha"], seed=mseed)
        m_out, m_ext = train_one(mcfg, dom, pts, device, grid)
        spent += m_out["wall_time"]
        restarts.append(
            dict(
                seed=mseed,
                L_total=m_out["L_total"],
                E_test=m_out["E_test"],
                E_val=m_out["E_val"],
                wall=m_out["wall_time"],
                stop=m_out["lbfgs_stop"],
            )
        )
        if best is None or m_out["L_total"] < best["L_total"]:
            best, best_ext = dict(m_out, restart=r), m_ext
    res = dict(
        rbf_E_test=r_out["E_test"],
        rbf_E_val=r_out["E_val"],
        rbf_time=budget,
        rbf_stop=r_out["lbfgs_stop"],
        rbf_L_total=r_out["L_total"],
        mlp_E_test=best["E_test"],
        mlp_E_val=best["E_val"],
        mlp_best_restart=best["restart"],
        mlp_n_restarts=len(restarts),
        mlp_time=spent,
        time_ratio=spent / budget,
        mlp_first_E_test=restarts[0]["E_test"],
        restarts=restarts,
    )
    extras = dict(
        rbf_err_t=r_ext["err_t"],
        mlp_err_t=best_ext["err_t"],
        t_eval=r_ext["t_eval"],
        rbf_abs_err_final=r_ext["abs_err_final"],
        mlp_abs_err_final=best_ext["abs_err_final"],
    )
    return res, extras


# ------------------------------------------------------------------ worker
def worker(w, nw, phase, hours):
    import torch

    smoke = phase.startswith("smoke")
    out_dir = "results_smoke" if smoke else RES
    os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
    if smoke:
        short = dict(adam_iters=50, lbfgs_iters=50)
        jobs = [dict(phase_jobs("A")[0], **short)]
        jobs += [dict(phase_jobs("B")[0], **short)]
        global BASE
        BASE = dict(BASE, **short)
    else:
        jobs = phase_jobs(phase)
    claims = os.path.join(out_dir, "claims")
    os.makedirs(claims, exist_ok=True)
    ng = torch.cuda.device_count()
    device = torch.device(f"cuda:{w % ng}" if ng else "cpu")

    def expected_s(j):
        """median wall time of finished runs of the same kind (None if unknown)"""
        rows = [
            r
            for r in read_runs(out_dir)
            if "error" not in r
            and r.get("scenario") == j["scenario"]
            and r.get("alpha") == j["alpha"]
            and r.get("centers") == j.get("centers")
        ]
        if j["scenario"] == "B":
            v = [r["rbf_time"] + r["mlp_time"] for r in rows]
        else:
            v = [r["wall_time"] for r in rows]
        return float(np.median(v)) if v else None

    done = {r["id"] for r in read_runs(out_dir) if "error" not in r}
    todo = [j for j in jobs if job_id(j) not in done]
    # dynamic shared queue (claim files) + longest-expected-first ordering
    todo.sort(key=lambda j: -(expected_s(j) or 0.0))
    log(f"[w{w}] phase {phase} device={device} open jobs (shared queue): {len(todo)}")
    dom = make_domain("butterfly")
    grid = plotting_grid(dom)
    t_start = time.time()
    for j in todo:
        jid = job_id(j)
        if jid in {r["id"] for r in read_runs(out_dir) if "error" not in r}:
            continue
        try:
            fd = os.open(os.path.join(claims, jid), os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            continue  # another worker is running it
        exp = expected_s(j)
        elapsed = (time.time() - t_start) / 3600
        if elapsed > hours or (exp is not None and elapsed + 1.1 * exp / 3600 > hours + 0.5):
            os.remove(os.path.join(claims, jid))
            log(
                f"[w{w}] skip {jid}: expected {0 if exp is None else exp/3600:.1f} h does not fit the remaining time"
            )
            continue
        jid = job_id(j)
        try:
            if j["scenario"] == "B":
                res, extras = run_B(j, dom, device, grid)
            else:
                res, extras = train_one(j, dom, point_sets(dom, "lhs", j["seed"]), device, grid)
        except Exception as e:
            traceback.print_exc()
            res, extras = dict(error=repr(e)[:300]), {}
            if "out of memory" in repr(e).lower():
                torch.cuda.empty_cache()
        np.savez_compressed(os.path.join(out_dir, "fields", jid + ".npz"), **extras)
        with open(os.path.join(out_dir, f"runs_w{w}.jsonl"), "a") as f:
            f.write(json.dumps({"id": jid, **j, **res}) + "\n")
        if j["scenario"] == "B" and "error" not in res:
            log(
                f"[w{w}] {jid} RBF={res['rbf_E_test']:.3e} ({res['rbf_time']:.0f}s) | MLP best-of-{res['mlp_n_restarts']}="
                f"{res['mlp_E_test']:.3e} ({res['mlp_time']:.0f}s, x{res['time_ratio']:.2f})"
            )
        else:
            log(
                f"[w{w}] {jid} Ehat={res.get('E_test', np.nan):.3e} {res.get('wall_time', 0):.0f}s "
                f"{res.get('lbfgs_stop', res.get('error', ''))}"
            )


# ------------------------------------------------------------------ reports
def reports(out="figures"):
    import pandas as pd
    from scipy.stats import wilcoxon, rankdata
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    allrows = [r for r in read_runs(RES) if "error" not in r]
    by_id = {r["id"]: r for r in allrows}
    seeds = S["seeds"][:5]

    def paired_test(d):
        d = np.asarray(d, float)
        if len(d) < 2 or np.all(d == 0):
            return np.nan, np.nan
        rk = rankdata(np.abs(d))
        rb = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
        return float(wilcoxon(d).pvalue), float(rb)

    # ===================== Scenario A
    rowsA, testsA = [], []
    for c in SEL:
        C = matched_centers(c)
        pairs = []
        for a in ALPHAS:
            cell = []
            for s in seeds:
                rj = dict(
                    BASE, scenario="A", arch="rbf", kernel="gaussian", centers=C, alpha=a, seed=s
                )
                r = by_id.get(job_id(rj))
                m = by_id.get(mlp_stage1_id(c, a, s))
                if r and m:
                    cell.append((m["E_test"], r["E_test"], m["wall_time"], r["wall_time"]))
            if cell:
                arr = np.array(cell)
                rowsA.append(
                    dict(
                        budget=n_mlp_params(c),
                        mlp=ckey(c),
                        rbf_centers=C,
                        alpha=a,
                        n=len(arr),
                        mlp_mean=arr[:, 0].mean(),
                        mlp_std=arr[:, 0].std(ddof=1) if len(arr) > 1 else np.nan,
                        rbf_mean=arr[:, 1].mean(),
                        rbf_std=arr[:, 1].std(ddof=1) if len(arr) > 1 else np.nan,
                        advantage_mlp_over_rbf=arr[:, 0].mean() / arr[:, 1].mean(),
                        rbf_wins=int((arr[:, 1] < arr[:, 0]).sum()),
                        mlp_time=arr[:, 2].mean(),
                        rbf_time=arr[:, 3].mean(),
                        rbf_slowdown=arr[:, 3].mean() / arr[:, 2].mean(),
                    )
                )
                pairs += list(arr)
        if len(pairs) >= 2:
            arr = np.array(pairs)
            p, rb = paired_test(arr[:, 0] - arr[:, 1])
            testsA.append(
                dict(
                    budget=n_mlp_params(c),
                    mlp=ckey(c),
                    rbf_centers=C,
                    n_pairs=len(arr),
                    rbf_wins=int((arr[:, 1] < arr[:, 0]).sum()),
                    p_wilcoxon=p,
                    rank_biserial=rb,
                    geo_ratio_mlp_over_rbf=float(np.exp(np.mean(np.log(arr[:, 0] / arr[:, 1])))),
                )
            )
    # per-alpha tests pooled over the three budgets (directions differ between alphas,
    # so pooling over alpha is not meaningful)
    testsA_alpha = []
    for a in ALPHAS:
        pairs = []
        for c in SEL:
            C = matched_centers(c)
            for s in seeds:
                rj = dict(
                    BASE, scenario="A", arch="rbf", kernel="gaussian", centers=C, alpha=a, seed=s
                )
                r = by_id.get(job_id(rj))
                m = by_id.get(mlp_stage1_id(c, a, s))
                if r and m:
                    pairs.append((m["E_test"], r["E_test"]))
        if len(pairs) >= 2:
            arr = np.array(pairs)
            p, rb = paired_test(arr[:, 0] - arr[:, 1])
            testsA_alpha.append(
                dict(
                    alpha=a,
                    n_pairs=len(arr),
                    rbf_wins=int((arr[:, 1] < arr[:, 0]).sum()),
                    p_wilcoxon=p,
                    rank_biserial=rb,
                    geo_ratio_mlp_over_rbf=float(np.exp(np.mean(np.log(arr[:, 0] / arr[:, 1])))),
                )
            )
    TAa = pd.DataFrame(testsA_alpha)
    TAa.to_csv(f"{out}/scenarioA_tests_by_alpha.csv", index=False)
    if len(TAa):
        log(
            "SCENARIO A paired tests per alpha (pooled over budgets):\n"
            + TAa.round(5).to_string(index=False)
        )
    A = pd.DataFrame(rowsA)
    TA = pd.DataFrame(testsA)
    A.to_csv(f"{out}/scenarioA_table.csv", index=False)
    TA.to_csv(f"{out}/scenarioA_tests.csv", index=False)
    if len(A):
        log("SCENARIO A (parameter-matched):\n" + A.round(5).to_string(index=False))
        log(
            "SCENARIO A paired tests pooled over alpha (NOT meaningful when directions differ):\n"
            + TA.round(5).to_string(index=False)
        )
        budgets = sorted(A.budget.unique())
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
        for ax, col, title, cmap in [
            (axs[0], "rbf_mean", "RBF-fPINN: mean test error", "Reds_r"),
            (axs[1], "mlp_mean", "MLP-fPINN: mean test error", "Reds_r"),
            (axs[2], "advantage_mlp_over_rbf", "MLP / RBF error ratio (>1: RBF better)", "RdBu_r"),
        ]:
            M = np.full((3, len(budgets)), np.nan)
            for _, rr in A.iterrows():
                M[ALPHAS.index(rr.alpha), budgets.index(rr.budget)] = rr[col]
            if col == "advantage_mlp_over_rbf":
                lim = np.nanmax(np.abs(np.log10(M))) if np.isfinite(M).any() else 1
                im = ax.imshow(np.log10(M), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
                txt = lambda v: f"{v:.2f}x"
            else:
                im = ax.imshow(M, cmap=cmap, aspect="auto")
                txt = lambda v: f"{v:.2e}"
            for i in range(3):
                for k in range(len(budgets)):
                    if np.isfinite(M[i, k]):
                        ax.text(k, i, txt(M[i, k]), ha="center", va="center", fontsize=9)
            ax.set_xticks(range(len(budgets)))
            ax.set_xticklabels(budgets)
            ax.set_xlabel("parameter budget")
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"alpha={a}" for a in ALPHAS])
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{out}/scenarioA_heatmaps.png", dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        wdt = 0.13
        for i, b in enumerate(budgets):
            for jx, a in enumerate(ALPHAS):
                rr = A[(A.budget == b) & (A.alpha == a)]
                if len(rr):
                    x = jx + (i - 1) * 2 * wdt
                    ax.bar(x - wdt / 2, rr.rbf_time.iloc[0], wdt, color=f"C{i}")
                    ax.bar(
                        x + wdt / 2, rr.mlp_time.iloc[0], wdt, color=f"C{i}", hatch="//", alpha=0.5
                    )
                    ax.text(
                        x,
                        rr.rbf_time.iloc[0],
                        f"{rr.rbf_slowdown.iloc[0]:.1f}x",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"alpha={a}" for a in ALPHAS])
        ax.set_ylabel("mean wall time (s)")
        ax.set_title(
            "Scenario A: RBF (solid) vs MLP (hatched) at matched budgets "
            + str(budgets)
            + "; label = RBF/MLP time"
        )
        fig.tight_layout()
        fig.savefig(f"{out}/scenarioA_time.png", dpi=200)
        plt.close(fig)

    # ===================== Scenario B
    B = [r for r in allrows if r.get("scenario") == "B"]
    if B:
        dB = pd.DataFrame(B)
        rowsB = []
        for a in ALPHAS:
            d = dB[dB.alpha == a]
            if not len(d):
                continue
            p, rb = paired_test(d.mlp_E_test - d.rbf_E_test)
            rowsB.append(
                dict(
                    alpha=a,
                    n=len(d),
                    rbf_mean=d.rbf_E_test.mean(),
                    mlp_mean=d.mlp_E_test.mean(),
                    mlp_single_run_mean=d.mlp_first_E_test.mean(),
                    ratio_mlp_over_rbf=d.mlp_E_test.mean() / d.rbf_E_test.mean(),
                    rbf_wins=int((d.rbf_E_test < d.mlp_E_test).sum()),
                    p_wilcoxon=p,
                    rank_biserial=rb,
                    rbf_time=d.rbf_time.mean(),
                    mean_restarts=d.mlp_n_restarts.mean(),
                    time_per_restart=(d.mlp_time / d.mlp_n_restarts).mean(),
                    mlp_time_overshoot=d.time_ratio.mean() - 1,
                )
            )
        p, rb = paired_test(dB.mlp_E_test - dB.rbf_E_test)
        rowsB.append(
            dict(
                alpha="all",
                n=len(dB),
                rbf_mean=dB.rbf_E_test.mean(),
                mlp_mean=dB.mlp_E_test.mean(),
                mlp_single_run_mean=dB.mlp_first_E_test.mean(),
                ratio_mlp_over_rbf=dB.mlp_E_test.mean() / dB.rbf_E_test.mean(),
                rbf_wins=int((dB.rbf_E_test < dB.mlp_E_test).sum()),
                p_wilcoxon=p,
                rank_biserial=rb,
                rbf_time=dB.rbf_time.mean(),
                mean_restarts=dB.mlp_n_restarts.mean(),
                time_per_restart=(dB.mlp_time / dB.mlp_n_restarts).mean(),
                mlp_time_overshoot=dB.time_ratio.mean() - 1,
            )
        )
        TB = pd.DataFrame(rowsB)
        TB.to_csv(f"{out}/scenarioB_tests.csv", index=False)
        log("SCENARIO B (time-matched):\n" + TB.round(5).to_string(index=False))
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(len(ALPHAS))
        for lab, col, mk in [
            ("RBF-fPINN (140 centers, single run)", "rbf_E_test", "o"),
            ("MLP-fPINN, time-matched best restart", "mlp_E_test", "s"),
            ("MLP-fPINN, single run (restart 0)", "mlp_first_E_test", "^"),
        ]:
            m = [dB[dB.alpha == a][col].mean() for a in ALPHAS]
            sd = [dB[dB.alpha == a][col].std() for a in ALPHAS]
            axs[0].errorbar(x, m, yerr=sd, marker=mk, capsize=3, label=lab)
        axs[0].set_xticks(x)
        axs[0].set_xticklabels([f"alpha={a}" for a in ALPHAS])
        axs[0].set_yscale("log")
        axs[0].set_ylabel("test error (mean +- std)")
        axs[0].legend(fontsize=8)
        axs[0].set_title("Scenario B: time-matched comparison")
        rat = [
            TB[TB.alpha == a].ratio_mlp_over_rbf.iloc[0] if (TB.alpha == a).any() else np.nan
            for a in ALPHAS
        ]
        axs[1].bar(x, rat, color="tab:purple")
        axs[1].axhline(1, ls="--", c="gray")
        for xi, v in zip(x, rat):
            axs[1].text(xi, v, f"{v:.2f}x", ha="center", va="bottom")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels([f"alpha={a}" for a in ALPHAS])
        axs[1].set_ylabel("MLP / RBF error ratio")
        axs[1].set_title("> 1: RBF better under equal time")
        fig.tight_layout()
        fig.savefig(f"{out}/scenarioB.png", dpi=200)
        plt.close(fig)

        # ===================== time-resolved error E(t) (from the Scenario-B runs)
        curves = {a: dict(rbf=[], mlp=[]) for a in ALPHAS}
        t_eval = None
        for r in B:
            fn = f"{RES}/fields/{r['id']}.npz"
            if os.path.exists(fn):
                z = np.load(fn)
                if "rbf_err_t" in z:
                    curves[r["alpha"]]["rbf"].append(z["rbf_err_t"])
                    curves[r["alpha"]]["mlp"].append(z["mlp_err_t"])
                    t_eval = z["t_eval"]
        if t_eval is not None:
            fig, axs = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True)
            rowsC = []
            for jx, a in enumerate(ALPHAS):
                if not curves[a]["rbf"]:
                    continue
                R = np.median(np.array(curves[a]["rbf"]), 0)
                Mm = np.median(np.array(curves[a]["mlp"]), 0)
                axs[0, jx].semilogy(t_eval, R, label="RBF-fPINN")
                axs[0, jx].semilogy(t_eval, Mm, label="MLP-fPINN")
                axs[0, jx].set_title(f"alpha = {a}: relative error E(t) (median over seeds)")
                axs[0, jx].legend(fontsize=8)
                axs[1, jx].semilogy(t_eval, Mm / np.maximum(R, 1e-12), c="tab:purple")
                axs[1, jx].axhline(1, ls="--", c="gray")
                axs[1, jx].set_title("MLP / RBF error ratio over time")
                axs[1, jx].set_xlabel("t")
                early = t_eval <= 0.1
                rowsC.append(
                    dict(
                        alpha=a,
                        rbf_E_early=R[early].mean(),
                        mlp_E_early=Mm[early].mean(),
                        ratio_early=(Mm[early] / R[early]).mean(),
                        rbf_E_late=R[~early].mean(),
                        mlp_E_late=Mm[~early].mean(),
                        ratio_late=(Mm[~early] / R[~early]).mean(),
                    )
                )
            fig.tight_layout()
            fig.savefig(f"{out}/error_vs_time.png", dpi=200)
            plt.close(fig)
            TC = pd.DataFrame(rowsC)
            TC.to_csv(f"{out}/error_vs_time.csv", index=False)
            log(
                "TIME-RESOLVED ERROR (early: t<=0.1, late: t>0.1):\n"
                + TC.round(5).to_string(index=False)
            )


# ------------------------------------------------------------------ orchestration
def restore():
    os.makedirs(os.path.join(RES, "fields"), exist_ok=True)
    dirs = sorted(
        {
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(PREV_RESULTS_DIR, "**", "runs_w*.jsonl"), recursive=True
            )
            if "results_smoke" not in p
        }
    )
    for i, d in enumerate(dirs):
        for fn in glob.glob(os.path.join(d, "runs_w*.jsonl")):
            shutil.copy(fn, os.path.join(RES, f"runs_wimp{i}_{os.path.basename(fn)}"))
        fd = os.path.join(d, "fields")
        if os.path.isdir(fd):
            for fn in os.listdir(fd):
                if fn.startswith("S2B_") or "_C" in fn:  # only Stage-2 fields are needed
                    if not os.path.exists(os.path.join(RES, "fields", fn)):
                        shutil.copy(os.path.join(fd, fn), os.path.join(RES, "fields", fn))
        log("imported", d)
    rows = read_runs(RES)
    n_mlp = sum(
        1
        for c in SEL
        for a in ALPHAS
        for s in S["seeds"][:5]
        if mlp_stage1_id(c, a, s) in {r["id"] for r in rows}
    )
    log(
        f"restored {len(rows)} runs | Stage-1 MLP runs needed for Scenario A found: {n_mlp}/45 | "
        f"Stage-2 runs: {sum(1 for r in rows if r.get('scenario') in ('A', 'B'))}"
    )
    if n_mlp < 45:
        log("WARNING: add the Stage-1 results dataset as input (needed for Scenario A)")


def offline_checks():
    for c in SEL:
        assert abs(5 * matched_centers(c) - n_mlp_params(c)) <= 3
    log(
        "parameter matching: "
        + ", ".join(
            f"{ckey(c)} {n_mlp_params(c)} <-> RBF {matched_centers(c)} centers ({5*matched_centers(c)})"
            for c in SEL
        )
    )
    log(
        f"OFFLINE CHECKS PASSED | Scenario B: {len(phase_jobs('B'))} jobs | Scenario A: {len(phase_jobs('A'))} RBF runs"
    )


def run_phase(phase, hours):
    import torch

    nw = max(1, torch.cuda.device_count())
    t0 = time.time()
    ids = {job_id(j) for j in phase_jobs(phase)}
    procs = [
        subprocess.Popen(
            [sys.executable, "-u", __file__, "worker", str(w), str(nw), phase, str(hours)]
        )
        for w in range(nw)
    ]
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
        log(f"scenario {phase}: {n}/{len(ids)} | {(time.time() - t0)/3600:.1f} h")
        if time.time() - t0 > (hours + 0.8) * 3600:
            for p in procs:
                p.terminate()
            log("hard stop")
            break


def main_all():
    import torch

    T0 = time.time()
    restore()
    log(
        "GPUs:",
        torch.cuda.device_count(),
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )
    offline_checks()
    shutil.rmtree("results_smoke", ignore_errors=True)
    subprocess.run([sys.executable, "-u", __file__, "worker", "0", "1", "smoke", "1"], check=False)
    sm = read_runs("results_smoke")
    bad = [r["id"] + ": " + r.get("error", "") for r in sm if "error" in r]
    if len(sm) != 2 or bad:
        raise SystemExit(f"SMOKE TEST FAILED -> nothing started: {bad or 'runs missing'}")
    log("SMOKE TEST PASSED (Scenario A RBF run + Scenario B RBF/MLP-restart job)")
    left = lambda: S["hours"] - (time.time() - T0) / 3600
    if S["run"]:
        for ph in ["B", "A"]:
            ids = {job_id(j) for j in phase_jobs(ph)}
            n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
            if n < len(ids) and left() > 0.4:
                run_phase(ph, left())
            n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
            log(
                f"scenario {ph}: {n}/{len(ids)}"
                + ("  -> COMPLETE" if n == len(ids) else "  -> continue in the next session")
            )
    try:
        reports()
    except Exception:
        traceback.print_exc()
        log("report generation failed -- results are still saved")
    shutil.make_archive("example42_stage2_results", "zip", ".", RES)
    log("DONE. figures/ and example42_stage2_results.zip are in the working directory")


if __name__ == "__main__":
    if sys.argv[1] == "all":
        main_all()
    else:
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
