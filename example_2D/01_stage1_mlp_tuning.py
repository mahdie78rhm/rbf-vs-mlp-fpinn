"""Example 4.2 (2D) - Stage 1: fair tuning of the MLP-fPINN against the RBF-fPINN baseline.

============================================================================
Example 4.2 -- STAGE 1: fair tuning of MLP-fPINN against the RBF-fPINN baseline
Single program driven by settings.json. Multi-session, fully automatic:
Phase A (screening, N=80) -> automatic top-3 selection on a VALIDATION set
-> Phase B (refinement, 5 seeds, all alpha x N, + pentagram) -> reports.
python 01_stage1_mlp_tuning.py all  |  python 01_stage1_mlp_tuning.py worker W NW PHASE HOURS
============================================================================
"""

import sys, os, json, time, math, glob, itertools, traceback, subprocess, shutil
import numpy as np
from scipy.stats import qmc
from matplotlib.path import Path

SETTINGS = dict(
    # identical seed pool to the reproduction stage (runs are seed-paired with RBF)
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
    run=True,  # False = report only (no training): merges inputs and builds the MLP-vs-RBF reports
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


# ------------------------------------------------------------------ search space
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
)  # protocol identical to the RBF baseline


def space():
    cfgs = [
        dict(arch="mlp", depth=d, width=w, act=a)
        for d, w, a in itertools.product([1, 2, 3, 4], [50, 100, 150, 200], ["tanh", "sin"])
    ]
    return cfgs


def ckey(c):
    if c["arch"] == "mlp":
        return f"mlp{c['depth']}x{c['width']}_{c['act']}"
    return "rbf_" + c["kernel"]


def job_id(j):
    return (
        f"{j['domain']}_{ckey(j)}_{j['sampler']}_a{j['alpha']}_N{j['N']}_k{j['k']}"
        f"_{j['residual_form']}_ad{j['adam_iters']}_{j['dtype']}_s{j['seed']}"
    )


def phase_jobs(phase):
    if phase == "A":  # screening: alpha extremes, N = 80 (as in the paper's Figs. 7-9), 3 seeds
        return [
            dict(BASE, **c, alpha=a, N=80, seed=s)
            for s in S["seeds"][:3]
            for a in [0.9, 0.1]
            for c in space()
        ]
    sel = json.load(open(os.path.join(RES, "selection.json")))
    jobs = [
        dict(BASE, **c, alpha=a, N=N, seed=s)
        for s in S["seeds"][:5]
        for c in sel["configs"]
        for a in ALPHAS
        for N in [10, 40, 80]
    ]
    jobs += [
        dict(BASE, **sel["configs"][0], domain="pentagram", alpha=a, N=N, seed=s)
        for s in S["seeds"][:5]
        for a in ALPHAS
        for N in [10, 40, 80]
    ]
    return jobs


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
    XX, YY, mask = grid
    Zp = np.c_[XX[mask], YY[mask], np.full(mask.sum(), T)]
    err = np.abs(predict(Zp) - exact_np(Zp[:, 0], Zp[:, 1], Zp[:, 2], a, k)).astype(np.float32)
    return out, dict(loss_hist=np.array(hist, np.float32), abs_err_final=err)


# ------------------------------------------------------------------ worker
def worker(w, nw, phase, hours):
    import torch

    smoke = phase == "smoke"
    out_dir = "results_smoke" if smoke else RES
    os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
    if smoke:
        jobs = [
            dict(j, adam_iters=50, lbfgs_iters=50)
            for j in phase_jobs("A")
            if j["arch"] == "mlp" and j["depth"] == 2 and j["width"] == 50 and j["act"] == "tanh"
        ][:1]
    else:
        jobs = phase_jobs(phase)
    done = {r["id"] for r in read_runs(out_dir) if "error" not in r}
    mine = [j for i, j in enumerate(jobs) if i % nw == w and job_id(j) not in done]
    ng = torch.cuda.device_count()
    device = torch.device(f"cuda:{w % ng}" if ng else "cpu")
    log(f"[w{w}] phase {phase} device={device} jobs to do: {len(mine)}")
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
            f"[w{w}] {jid} Eval={res.get('E_val', np.nan):.3e} Ehat={res.get('E_test', np.nan):.3e} "
            f"{res.get('wall_time', 0):.0f}s {res.get('lbfgs_stop', res.get('error', ''))}"
        )


# ------------------------------------------------------------------ selection (pre-registered rule)
def selection(write=True):
    """Top-3 configurations of Phase A by the median of log10(E_val) over
    alpha in {0.9, 0.1} x 3 seeds. Ties (within 10%) -> fewer parameters.
    """
    ids = {job_id(j) for j in phase_jobs("A")}
    rows = [r for r in read_runs(RES) if r["id"] in ids]
    if len({r["id"] for r in rows}) < len(ids):
        return None
    stats = {}
    for r in rows:
        if "error" in r:
            continue
        stats.setdefault(
            ckey(r),
            dict(
                cfg={k: r[k] for k in ["arch", "depth", "width", "act"] if k in r},
                v=[],
                p=r["n_params"],
            ),
        )["v"].append(math.log10(r["E_val"]))
    table = sorted(
        ((np.median(s["v"]), s["p"], key, s["cfg"]) for key, s in stats.items() if len(s["v"]) >= 4)
    )
    remaining, chosen = list(table), []
    while remaining and len(chosen) < 3:
        best = remaining[0][0]
        near = [t for t in remaining if t[0] <= best + math.log10(1.1)]
        pick = min(near, key=lambda t: t[1])
        chosen.append(pick)
        remaining.remove(pick)
    sel = dict(
        rule="median log10 E_val over Phase A (alpha 0.9, 0.1; N=80; 3 seeds); ties within 10% -> fewer params",
        keys=[c[2] for c in chosen],
        configs=[c[3] for c in chosen],
        median_val=[10 ** c[0] for c in chosen],
        n_params=[c[1] for c in chosen],
    )
    if write:
        json.dump(sel, open(os.path.join(RES, "selection.json"), "w"), indent=1)
    return sel


# ------------------------------------------------------------------ reports
def reports(out="figures"):
    import pandas as pd
    from scipy.stats import wilcoxon, rankdata
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    rows = [r for r in read_runs(RES) if "error" not in r]
    if not rows:
        log("no results")
        return
    df = pd.DataFrame(rows)
    for c in ["kernel", "arch", "sampler", "residual_form", "adam_iters", "dtype", "k"]:
        if c not in df:
            df[c] = None
    df["cfg"] = [ckey(r) for r in rows]
    main = (
        (df.residual_form == "scaled")
        & (df.adam_iters == 5000)
        & (df.dtype == "float32")
        & (df.k == 1)
    )
    rbf = df[main & (df.arch == "rbf") & (df.kernel == "gaussian") & (df.sampler == "lhs")]
    if rbf.empty:
        log(
            "WARNING: no RBF baseline runs found -> MLP-only reports (add the reproduction results as input "
            "to get the paired MLP-vs-RBF comparison)"
        )
    mlp = df[main & (df.arch == "mlp")]

    # failure flag: error > 3x the median of its (config, domain, alpha, N) group
    def flag(d):
        med = d.groupby(["cfg", "domain", "alpha", "N"]).E_test.transform("median")
        return d.E_test > 3 * med

    mlp = mlp.assign(failed=flag(mlp))

    # ---- Phase A ranking
    idsA = {job_id(j) for j in phase_jobs("A")}
    A = mlp[mlp.id.isin(idsA)]
    if len(A):
        g = (
            A.groupby("cfg")
            .agg(
                arch=("arch", "first"),
                n_params=("n_params", "first"),
                n=("E_val", "size"),
                val_median=("E_val", "median"),
                val_mean=("E_val", "mean"),
                test_median=("E_test", "median"),
                time_mean=("wall_time", "mean"),
                fail_rate=("failed", "mean"),
            )
            .reset_index()
        )
        for a in [0.9, 0.1]:
            g[f"val_median_a{a}"] = g.cfg.map(A[A.alpha == a].groupby("cfg").E_val.median())
        g = g.sort_values("val_median")
        g.to_csv(f"{out}/phaseA_ranking.csv", index=False)
        # sensitivity: which top-3 would the Example-4.1 rule (mean TEST error) have picked?
        g41 = A.groupby("cfg").E_test.mean().sort_values()
        sel_file = os.path.join(RES, "selection.json")
        top_ours = (
            json.load(open(sel_file))["keys"][:3] if os.path.exists(sel_file) else list(g.cfg[:3])
        )
        top_41 = list(g41.index[:3])
        json.dump(
            dict(
                rule_ours="median log10 E_val, ties within 10% -> fewer params (actual selection)",
                top3_ours=top_ours,
                rule_example41="mean E_test (as in Example 4.1, Stage 1)",
                top3_example41=top_41,
                overlap=len(set(top_ours) & set(top_41)),
            ),
            open(f"{out}/selection_rule_sensitivity.json", "w"),
            indent=1,
        )
        log(
            f"SELECTION-RULE CHECK: ours {top_ours} | Example-4.1 rule {top_41} | overlap {len(set(top_ours) & set(top_41))}/3"
        )
        rA = rbf[
            (rbf.domain == "butterfly")
            & (rbf.N == 80)
            & rbf.alpha.isin([0.9, 0.1])
            & rbf.seed.isin(S["seeds"][:3])
        ]
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["tab:red" for a in g.arch]
        ax.bar(range(len(g)), g.test_median, color=colors)
        if len(rA):
            ax.axhline(
                rA.E_test.median(),
                color="tab:blue",
                ls="--",
                label=f"RBF-fPINN Gaussian (median {rA.E_test.median():.2e})",
            )
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.cfg, rotation=90, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("median test error (Phase A)")
        ax.set_title("Phase A: MLP configurations ranked by validation error ")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out}/phaseA_ranking.png", dpi=200)
        plt.close(fig)
        log(
            "PHASE A top 8:\n"
            + g.head(8)[
                ["cfg", "n_params", "val_median", "test_median", "fail_rate", "time_mean"]
            ].to_string(index=False)
        )

    if not os.path.exists(os.path.join(RES, "selection.json")):
        return
    sel = json.load(open(os.path.join(RES, "selection.json")))
    idsB = {job_id(j) for j in phase_jobs("B")}
    B = mlp[mlp.id.isin(idsB | idsA)]
    B = B[B.seed.isin(S["seeds"][:5]) & B.N.isin([10, 40, 80])]
    summ, tests = [], []
    for key in sel["keys"]:
        for dom in ["butterfly", "pentagram"]:
            for a in ALPHAS:
                pairs = []
                for N in [10, 40, 80]:
                    m = (
                        B[(B.cfg == key) & (B.domain == dom) & (B.alpha == a) & (B.N == N)]
                        .set_index("seed")
                        .E_test
                    )
                    r = (
                        rbf[(rbf.domain == dom) & (rbf.alpha == a) & (rbf.N == N)]
                        .set_index("seed")
                        .E_test
                    )
                    if len(r) == 0:
                        r = pd.Series(dtype=float)
                    common = m.index.intersection(r.index)
                    if len(m) == 0:
                        continue
                    summ.append(
                        dict(
                            cfg=key,
                            domain=dom,
                            alpha=a,
                            N=N,
                            n=len(common),
                            mlp_mean=m.mean(),
                            mlp_std=m.std(),
                            mlp_median=m.median(),
                            rbf_mean=r.mean() if len(r) else np.nan,
                            rbf_std=r.std() if len(r) > 1 else np.nan,
                            ratio_mlp_over_rbf=m.mean() / r.mean() if len(r) else np.nan,
                            rbf_wins=int((r[common] < m[common]).sum()),
                        )
                    )
                    pairs += [(m[s], r[s]) for s in common]
                if len(pairs) >= 6:
                    d = np.array([p[0] - p[1] for p in pairs])
                    rk = rankdata(np.abs(d))
                    rb = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
                    tests.append(
                        dict(
                            cfg=key,
                            domain=dom,
                            alpha=a,
                            n_pairs=len(pairs),
                            rbf_wins=int((d > 0).sum()),
                            p_wilcoxon=wilcoxon(d).pvalue,
                            rank_biserial=rb,
                            geo_ratio_mlp_over_rbf=float(
                                np.exp(np.mean(np.log([p[0] / p[1] for p in pairs])))
                            ),
                        )
                    )
    pd.DataFrame(summ).to_csv(f"{out}/phaseB_summary.csv", index=False)
    T = pd.DataFrame(tests)
    T.to_csv(f"{out}/phaseB_tests.csv", index=False)
    if len(T):
        log(
            "PHASE B paired tests (MLP vs RBF, pooled over N):\n"
            + T.round(4).to_string(index=False)
        )

    # error vs N, per alpha (butterfly)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, a in zip(axs, ALPHAS):
        r = rbf[
            (rbf.domain == "butterfly")
            & (rbf.alpha == a)
            & rbf.N.isin([10, 40, 80])
            & rbf.seed.isin(S["seeds"][:5])
        ]
        if len(r):
            q = r.groupby("N").E_test.quantile([0.25, 0.5, 0.75]).unstack()
            ax.errorbar(
                q.index,
                q[0.5],
                yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]],
                marker="o",
                lw=2.5,
                color="tab:blue",
                capsize=3,
                label="RBF-fPINN (Gaussian)",
            )
        for key in sel["keys"]:
            m = B[(B.cfg == key) & (B.domain == "butterfly") & (B.alpha == a)]
            if len(m):
                q = m.groupby("N").E_test.quantile([0.25, 0.5, 0.75]).unstack()
                ax.errorbar(
                    q.index,
                    q[0.5],
                    yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]],
                    marker="s",
                    capsize=3,
                    label=key,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_title(f"butterfly, alpha = {a}")
    axs[0].set_ylabel("test error (median, IQR)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/phaseB_error_vs_N.png", dpi=200)
    plt.close(fig)

    # every seed, N = 80 (shows bimodality / failures)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, a in zip(axs, ALPHAS):
        labels = ["RBF"] + sel["keys"]
        for i, key in enumerate(labels):
            d = (
                rbf[
                    (rbf.domain == "butterfly")
                    & (rbf.alpha == a)
                    & (rbf.N == 80)
                    & rbf.seed.isin(S["seeds"][:5])
                ]
                if key == "RBF"
                else B[(B.cfg == key) & (B.domain == "butterfly") & (B.alpha == a) & (B.N == 80)]
            )
            fl = d["failed"].values if "failed" in d else np.zeros(len(d), bool)
            x = i + np.random.default_rng(i).uniform(-0.12, 0.12, len(d))
            ax.scatter(x[~fl], d.E_test.values[~fl], c="k", s=18)
            ax.scatter(x[fl], d.E_test.values[fl], c="r", marker="x", s=40)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"N = 80, alpha = {a} (red x = failed run)")
    axs[0].set_ylabel("test error, every seed")
    fig.tight_layout()
    fig.savefig(f"{out}/phaseB_seeds_N80.png", dpi=200)
    plt.close(fig)

    # error maps: winner vs RBF, butterfly, N = 80, first seed
    dom = make_domain("butterfly")
    XX, YY, mask = plotting_grid(dom)
    s0 = S["seeds"][0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for i, (name, d) in enumerate(
        [("RBF-fPINN (Gaussian)", rbf), (sel["keys"][0], B[B.cfg == sel["keys"][0]])]
    ):
        for jx, a in enumerate(ALPHAS):
            ax = axs[i, jx]
            rr = d[(d.domain == "butterfly") & (d.alpha == a) & (d.N == 80) & (d.seed == s0)]
            fn = f"{RES}/fields/{rr.iloc[0]['id']}.npz" if len(rr) else ""
            if not os.path.exists(fn):
                ax.axis("off")
                continue
            Z = np.full(XX.shape, np.nan)
            Z[mask] = np.load(fn)["abs_err_final"]
            pc = ax.pcolormesh(XX, YY, Z, cmap="jet", shading="auto", vmin=0)
            pp = np.vstack([dom.poly, dom.poly[:1]])
            ax.plot(pp[:, 0], pp[:, 1], "k", lw=1.2)
            ax.set_aspect("equal")
            ax.set_title(f"{name}, alpha = {a}", fontsize=10)
            plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Absolute error at final time, N = 80")
    fig.tight_layout()
    fig.savefig(f"{out}/phaseB_error_maps.png", dpi=200)
    plt.close(fig)
    mlp.groupby(["cfg", "domain"]).agg(
        n=("seed", "size"),
        fail_rate=("failed", "mean"),
        stall=("lbfgs_stop", lambda s: (s == "change_tol_or_stall").mean()),
        iters=("lbfgs_n_iter", "mean"),
        time=("wall_time", "mean"),
    ).reset_index().to_csv(f"{out}/diagnostics.csv", index=False)


# ------------------------------------------------------------------ orchestration
def restore():
    """Merge the results of EVERY input dataset (RBF baseline + earlier Stage-1 sessions)."""
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
                if not os.path.exists(os.path.join(RES, "fields", fn)):
                    shutil.copy(os.path.join(fd, fn), os.path.join(RES, "fields", fn))
        log("imported", d)
    rows = read_runs(RES)
    log(
        f"restored {len(rows)} runs  (RBF: {sum(r.get('arch') == 'rbf' for r in rows)}, "
        f"MLP: {sum(r.get('arch') == 'mlp' for r in rows)})"
    )
    if not any(r.get("arch") == "rbf" for r in rows):
        log(
            "WARNING: RBF baseline results not found -> add the 'example42-results' dataset as input for the comparison"
        )


def offline_checks():
    from scipy.integrate import quad

    for a in ALPHAS:
        t, x, y = 0.7, 0.3, -0.4
        gp = lambda s: s ** (1 - a) / math.gamma(2 - a)
        cap = quad(lambda s: (t - s) ** (-a) * gp(s), 0, t, limit=200)[0] / math.gamma(1 - a)
        Sx = np.sin(np.pi * x) * np.sin(np.pi * y)
        assert (
            abs(
                gp(t) * Sx
                + cap * Sx
                + 2 * np.pi**2 * time_factor(t, a) * Sx
                - source_np(x, y, t, a)
            )
            < 1e-6
        )
        assert np.allclose(l1_matrix(a, 80)[0].sum(1), 0)
    for name in ["butterfly", "pentagram"]:
        dom = make_domain(name)
        xv, tv = validation_set(dom)
        assert dom.inside(xv[:300]).all() and len(xv) == 400 and len(tv) == 20
    log(
        f"OFFLINE CHECKS PASSED | Phase A: {len(space())} configurations x 2 alpha x 3 seeds = {len(phase_jobs('A'))} runs"
    )


def run_phase(phase, hours):
    import torch

    nw = max(1, torch.cuda.device_count())
    t0 = time.time()
    total = len(phase_jobs(phase))
    procs = [
        subprocess.Popen(
            [sys.executable, "-u", __file__, "worker", str(w), str(nw), phase, str(hours)]
        )
        for w in range(nw)
    ]
    ids = {job_id(j) for j in phase_jobs(phase)}
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
        log(f"phase {phase}: {n}/{total} runs | {(time.time() - t0)/3600:.1f} h")
        if time.time() - t0 > (hours + 0.7) * 3600:
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
    bad = [
        r["id"] + ": " + r.get("error", "check failed")
        for r in sm
        if "error" in r
        or not (r["exact_L_ic"] < 1e-10 and r["exact_L_bc"] < 1e-10 and np.isfinite(r["E_val"]))
    ]
    if len(sm) != 1 or bad:
        raise SystemExit(f"SMOKE TEST FAILED -> nothing started: {bad or 'runs missing'}")
    log("SMOKE TEST PASSED")
    left = lambda: S["hours"] - (time.time() - T0) / 3600
    if S["run"]:
        if selection(write=False) is None:
            run_phase("A", left())
        sel = selection()
        if sel is None:
            log("Phase A not finished yet -> it will continue automatically in the next session")
        else:
            log(
                "SELECTED for Phase B: "
                + ", ".join(
                    f"{k} ({p} params, val {v:.2e})"
                    for k, p, v in zip(sel["keys"], sel["n_params"], sel["median_val"])
                )
            )
            if left() > 0.4:
                run_phase("B", left())
            ids = {job_id(j) for j in phase_jobs("B")}
            n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
            log(
                f"Phase B: {n}/{len(ids)} runs"
                + ("  -> COMPLETE" if n == len(ids) else "  -> continue in the next session")
            )
    if not os.path.exists(os.path.join(RES, "selection.json")):
        if selection() is not None:  # report-only runs: rebuild the (deterministic) selection first
            log(
                "selection rebuilt from Phase A results: "
                + ", ".join(json.load(open(os.path.join(RES, "selection.json")))["keys"])
            )
    try:
        reports()
    except Exception:
        traceback.print_exc()
        log("report generation failed -- results are still saved")
    shutil.make_archive("example42_stage1_results", "zip", ".", RES)
    log("DONE. figures/ and example42_stage1_results.zip are in the working directory")


if __name__ == "__main__":
    if sys.argv[1] == "all":
        main_all()
    else:
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
