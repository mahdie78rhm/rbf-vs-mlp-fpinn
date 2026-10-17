"""Example 4.2 (2D) - Stage 3: confirmation at the reference setting (k = 1).

RBF-fPINN (Gaussian, 140 centers) versus the Stage-1 winner MLP-fPINN (2 x 150, tanh) on the
butterfly domain with LHS points, N = 80, alpha in {0.9, 0.1} and the 10 pre-registered seeds
(40 runs). The exact solution is the one of the reference paper,
u = (1 + t^(2-alpha)/Gamma(3-alpha)) sin(pi x) sin(pi y). Besides the test error over all times
the final-time error E(T) on a 301 x 301 grid and the time-resolved error E(t) are recorded.
The first five seeds repeat configurations of Stages 0-1 (identical run ids) and are re-used
when the earlier results are found under PREV_RESULTS_DIR.

    python 03_stage3_confirmation_k1.py all
    python 03_stage3_confirmation_k1.py worker W NW PHASE HOURS
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

ALPHAS = [0.9, 0.1]  # the two regimes where the winner differs (Stages 1-2)
KS = [1]  # k = 1: the reference problem u = g(t) sin(pi x) sin(pi y)
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
    u_fine = exact_np(Zp[:, 0], Zp[:, 1], Zp[:, 2], a, k)
    p_fine = predict(Zp)
    err = np.abs(p_fine - u_fine).astype(np.float32)
    out["E_fine_final"] = float(np.linalg.norm(p_fine - u_fine) / np.linalg.norm(u_fine))
    return out, dict(
        loss_hist=np.array(hist, np.float32),
        abs_err_final=err,
        err_t=err_t,
        t_eval=t_eval.astype(np.float32),
    )


# ------------------------------------------------------------------ protocol
BASE = dict(
    T=1.0,
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
ARMS = {
    "rbf": dict(arch="rbf", kernel="gaussian", centers=140),
    "mlp": dict(arch="mlp", depth=2, width=150, act="tanh"),
}
ARM_LABEL = {"rbf": "RBF-fPINN (140 centers)", "mlp": "MLP-fPINN (2x150 tanh)"}
PHASES = {"core": ["rbf", "mlp"]}


def ckey(c):
    if c["arch"] == "mlp":
        return f"mlp{c['depth']}x{c['width']}_{c['act']}"
    return "gaussian" if c.get("centers", 140) == 140 else f"gaussian_C{c['centers']}"


def job_id(j):  # identical scheme to the reproduction (RBF) and Stage-1 (MLP) runs
    return (
        f"{j['domain']}_{ckey(j)}_{j['sampler']}_a{j['alpha']}_N{j['N']}_k{j['k']}"
        f"_{j['residual_form']}_ad{j['adam_iters']}_{j['dtype']}_s{j['seed']}"
    )


def make_job(arm, k, a, s):
    return dict(BASE, **ARMS[arm], arm=arm, k=k, alpha=a, seed=s)


def phase_jobs(phase):
    return [
        make_job(arm, k, a, s)
        for arm in PHASES[phase]
        for s in S["seeds"][:10]
        for k in KS
        for a in ALPHAS
    ]


def all_jobs():
    return phase_jobs("core")


def arm_of(r):
    for name, c in ARMS.items():
        if r.get("arch") == c["arch"] and ckey(r) == ckey(c):
            return name
    return None


# ------------------------------------------------------------------ worker
def worker(w, nw, phase, hours):
    import torch

    smoke = phase.startswith("smoke")
    out_dir = "results_smoke" if smoke else RES
    os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
    if smoke:
        jobs = [
            dict(make_job(arm, 1, 0.9, S["seeds"][0]), adam_iters=50, lbfgs_iters=50)
            for arm in ARMS
        ]
    else:
        jobs = phase_jobs(phase)
    claims = os.path.join(out_dir, "claims")
    os.makedirs(claims, exist_ok=True)
    ng = torch.cuda.device_count()
    device = torch.device(f"cuda:{w % ng}" if ng else "cpu")
    seeds = S["seeds"][:10]

    def expected_s(j, rows):
        """median wall time of finished runs of the same arm/alpha (same k if available)"""
        same = [
            r
            for r in rows
            if "error" not in r
            and arm_of(r) == j["arm"]
            and r.get("alpha") == j["alpha"]
            and r.get("N") == 80
            and r.get("adam_iters") == 5000
        ]
        v = [r["wall_time"] for r in same if r.get("k") == j["k"]] or [r["wall_time"] for r in same]
        return float(np.median(v)) if v else None

    rows = read_runs(out_dir)
    done = {r["id"] for r in rows if "error" not in r}
    todo = [j for j in jobs if job_id(j) not in done]
    # priority blocks: seeds 1-5 before seeds 6-10 (balanced partial results across k);
    # inside a block longest-expected-first (better packing on the two GPUs)
    todo.sort(key=lambda j: (seeds.index(j["seed"]) >= 5, -(expected_s(j, rows) or 0.0)))
    log(f"[w{w}] phase {phase} device={device} open jobs (shared queue): {len(todo)}")
    dom = make_domain("butterfly")
    grid = plotting_grid(dom)
    t_start = time.time()
    for j in todo:
        jid = job_id(j)
        rows = read_runs(out_dir)
        if jid in {r["id"] for r in rows if "error" not in r}:
            continue
        try:
            fd = os.open(os.path.join(claims, jid), os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            continue  # another worker is running it
        exp = expected_s(j, rows)
        elapsed = (time.time() - t_start) / 3600
        if elapsed > hours or (exp is not None and elapsed + 1.1 * exp / 3600 > hours + 0.5):
            os.remove(os.path.join(claims, jid))
            log(
                f"[w{w}] skip {jid}: expected {0 if exp is None else exp/60:.0f} min does not fit the remaining time"
            )
            continue
        try:
            res, extras = train_one(j, dom, point_sets(dom, "lhs", j["seed"]), device, grid)
        except Exception as e:
            traceback.print_exc()
            res, extras = dict(error=repr(e)[:300]), {}
            if "out of memory" in repr(e).lower():
                torch.cuda.empty_cache()
        np.savez_compressed(os.path.join(out_dir, "fields", jid + ".npz"), **extras)
        with open(os.path.join(out_dir, f"runs_w{w}.jsonl"), "a") as f:
            f.write(json.dumps({"id": jid, **j, **res}) + "\n")
        log(
            f"[w{w}] {j['arm']:>3} k={j['k']} a={j['alpha']} s={j['seed']}: Ehat={res.get('E_test', np.nan):.3e} "
            f"Efine={res.get('E_fine_final', np.nan):.3e} {res.get('wall_time', 0):.0f}s {res.get('lbfgs_stop', res.get('error', ''))}"
        )


# ------------------------------------------------------------------ reports
def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    m = len(idx)
    if m:
        order = idx[np.argsort(p[idx])]
        run = 0.0
        for i, j in enumerate(order):
            run = max(run, min(1.0, (m - i) * p[j]))
            out[j] = run
    return out


def e_fine(r, dom_grid):
    """final-time relative L2 error on the 301^2 plotting grid; computed from the stored
    |error| field for re-used runs that pre-date the E_fine_final metric"""
    if "E_fine_final" in r:
        return r["E_fine_final"]
    fn = f"{RES}/fields/{r['id']}.npz"
    if not os.path.exists(fn):
        return np.nan
    try:
        err = np.load(fn)["abs_err_final"]
    except Exception:
        return np.nan
    XX, YY, mask = dom_grid
    u = exact_np(XX[mask], YY[mask], 1.0, r["alpha"], r["k"])
    return float(np.linalg.norm(err) / np.linalg.norm(u)) if len(err) == len(u) else np.nan


def reports(out="figures"):
    import pandas as pd
    from scipy.stats import wilcoxon, rankdata
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    by_id = {r["id"]: r for r in read_runs(RES) if "error" not in r}
    dom = make_domain("butterfly")
    grid = plotting_grid(dom)
    seeds = S["seeds"][:10]
    col = {"rbf": "tab:blue", "mlp": "tab:red"}
    rec = []
    for arm in ARMS:
        for a in ALPHAS:
            for s in seeds:
                r = by_id.get(job_id(make_job(arm, 1, a, s)))
                if r:
                    rec.append(
                        dict(
                            arm=arm,
                            alpha=a,
                            seed=s,
                            E_test=r["E_test"],
                            E_val=r["E_val"],
                            E_fine=e_fine(r, grid),
                            time=r["wall_time"],
                            stop=r.get("lbfgs_stop", ""),
                            n_params=r.get("n_params", np.nan),
                            id=r["id"],
                        )
                    )
    D = pd.DataFrame(rec)
    if not len(D):
        log("no results yet")
        return
    D.to_csv(f"{out}/confirmation_all_runs.csv", index=False)

    # ---------------- summary table
    T = (
        D.groupby(["alpha", "arm"])
        .agg(
            n=("E_test", "size"),
            mean=("E_test", "mean"),
            std=("E_test", "std"),
            median=("E_test", "median"),
            fine_median=("E_fine", "median"),
            time_mean=("time", "mean"),
            n_params=("n_params", "first"),
        )
        .reset_index()
    )
    T.to_csv(f"{out}/confirmation_summary.csv", index=False)
    log(
        "CONFIRMATION SUMMARY (E_hat over all times; fine = final-time error E(T) on the 301^2 grid):\n"
        + T.to_string(index=False, float_format=lambda v: f"{v:.3e}")
    )

    # ---------------- paired tests RBF vs MLP per alpha (Holm over the alphas of each metric)
    def paired(metric):
        rows = []
        for a in ALPHAS:
            x = D[(D.arm == "rbf") & (D.alpha == a)].set_index("seed")[metric]
            y = D[(D.arm == "mlp") & (D.alpha == a)].set_index("seed")[metric]
            common = x.index.intersection(y.index)
            if len(common) < 2:
                continue
            d = (y[common] - x[common]).values  # > 0: RBF better
            p, rb = np.nan, np.nan
            if len(d) >= 5 and not np.all(d == 0):
                rk = rankdata(np.abs(d))
                rb = float((rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum())
                p = float(wilcoxon(d).pvalue)
            rows.append(
                dict(
                    metric=metric,
                    alpha=a,
                    n_pairs=len(common),
                    rbf_wins=int((d > 0).sum()),
                    geo_ratio_mlp_over_rbf=float(
                        np.exp(np.mean(np.log(y[common].values / x[common].values)))
                    ),
                    p_wilcoxon=p,
                    rank_biserial=rb,
                )
            )
        R = pd.DataFrame(rows)
        if len(R):
            R["p_holm"] = holm(R.p_wilcoxon.values)
        return R

    TT = pd.concat([paired("E_test"), paired("E_fine")], ignore_index=True)
    TT.to_csv(f"{out}/confirmation_tests.csv", index=False)
    if len(TT):
        log(
            "PAIRED TESTS RBF vs MLP  [ratio > 1: RBF more accurate]:\n"
            + TT.to_string(index=False, float_format=lambda v: f"{v:.4g}")
        )

    # ---------------- figure 1: paired MLP/RBF ratios (every seed) for E_hat and for the final-time error
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (metric, title) in zip(
        axs,
        [
            ("E_test", "test error over all times"),
            ("E_fine", "final-time error E(T), 301x301 grid"),
        ],
    ):
        for i, a in enumerate(ALPHAS):
            x = D[(D.arm == "rbf") & (D.alpha == a)].set_index("seed")[metric]
            y = D[(D.arm == "mlp") & (D.alpha == a)].set_index("seed")[metric]
            cm = x.index.intersection(y.index)
            if len(cm):
                r_ = y[cm].values / x[cm].values
                ax.scatter(
                    i + np.linspace(-0.12, 0.12, len(r_)),
                    r_,
                    s=18,
                    c=np.where(r_ > 1, "tab:blue", "tab:red"),
                    alpha=0.8,
                )
                gm = float(np.exp(np.mean(np.log(r_))))
                ax.hlines(gm, i - 0.3, i + 0.3, colors="k", lw=2)
                ax.text(i + 0.33, gm, f"{gm:.2f}", va="center", fontsize=9)
        ax.axhline(1, ls="--", c="gray")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f"alpha = {a}" for a in ALPHAS])
        ax.set_title(title)
    axs[0].set_ylabel("paired error ratio MLP / RBF (> 1: RBF more accurate)")
    fig.tight_layout()
    fig.savefig(f"{out}/confirmation_paired_ratios.png", dpi=200)
    plt.close(fig)

    # ---------------- figure 2: absolute error at t = T (first seed with both methods)
    XX, YY, mask = grid
    for a in ALPHAS:
        fields = {}
        for s in seeds:
            for arm in ARMS:
                fn = f"{RES}/fields/{job_id(make_job(arm, 1, a, s))}.npz"
                if os.path.exists(fn):
                    try:
                        z = np.load(fn)["abs_err_final"]
                        if len(z) == mask.sum() and arm not in fields:
                            fields[arm] = (s, z)
                    except Exception:
                        pass
            if len(fields) == len(ARMS):
                break
        if not fields:
            continue
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
        U = np.full(XX.shape, np.nan)
        U[mask] = exact_np(XX[mask], YY[mask], 1.0, a, 1)
        im = axs[0].pcolormesh(XX, YY, U, cmap="jet", shading="auto")
        fig.colorbar(im, ax=axs[0])
        axs[0].set_title(f"exact u(x, y, 1), alpha = {a}")
        vmax = max(f[1].max() for f in fields.values())
        for ax, arm in zip(axs[1:], ARMS):
            if arm in fields:
                E = np.full(XX.shape, np.nan)
                E[mask] = fields[arm][1]
                im = ax.pcolormesh(XX, YY, E, cmap="jet", shading="auto", vmin=0, vmax=vmax)
                fig.colorbar(im, ax=ax)
                ax.set_title(f"|error| {ARM_LABEL[arm]} (seed {fields[arm][0]})", fontsize=9)
        for ax in axs:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{out}/confirmation_error_maps_a{a}.png", dpi=170)
        plt.close(fig)

    # ---------------- figure 3: time-resolved error E(t), median over seeds, and the MLP/RBF ratio
    fig, axs = plt.subplots(
        2, len(ALPHAS), figsize=(5.5 * len(ALPHAS), 7), sharex=True, squeeze=False
    )
    rowsT = []
    for j, a in enumerate(ALPHAS):
        cur = {}
        for arm in ARMS:
            L = []
            for s in seeds:
                fn = f"{RES}/fields/{job_id(make_job(arm, 1, a, s))}.npz"
                if os.path.exists(fn):
                    try:
                        z = np.load(fn)
                        if "err_t" in z:
                            L.append(z["err_t"])
                            te = z["t_eval"]
                    except Exception:
                        pass
            if L:
                cur[arm] = np.median(np.array(L), 0)
                axs[0, j].semilogy(te, cur[arm], c=col[arm], label=ARM_LABEL[arm])
        axs[0, j].set_title(f"alpha = {a}: E(t), median over seeds")
        axs[0, j].grid(alpha=0.3, which="both")
        if "rbf" in cur and "mlp" in cur:
            ratio = cur["mlp"] / cur["rbf"]
            axs[1, j].semilogy(te, ratio, c="tab:purple")
            axs[1, j].axhline(1, ls="--", c="gray")
            axs[1, j].axvspan(
                0, 0.1, color="tab:orange", alpha=0.15, label="initial layer t <= 0.1"
            )
            axs[1, j].set_title("MLP / RBF error ratio over time")
            axs[1, j].set_xlabel("t")
            axs[1, j].grid(alpha=0.3, which="both")
            early = te <= 0.1
            rowsT.append(
                dict(
                    alpha=a,
                    n_seeds_with_E_t=len(L),
                    ratio_layer=float(np.mean(ratio[early])),
                    ratio_rest=float(np.mean(ratio[~early])),
                )
            )
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/confirmation_error_vs_time.png", dpi=170)
    plt.close(fig)
    TE = pd.DataFrame(rowsT)
    TE.to_csv(f"{out}/confirmation_time_resolved.csv", index=False)
    if len(TE):
        log(
            "MLP/RBF ratio inside the initial layer (t <= 0.1) vs the rest (t > 0.1):\n"
            + TE.to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )


# ------------------------------------------------------------------ orchestration
def restore():
    os.makedirs(os.path.join(RES, "fields"), exist_ok=True)
    need = {job_id(j) for j in all_jobs()}
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
            keep = []
            for line in open(fn):
                try:
                    if json.loads(line)["id"] in need:
                        keep.append(line if line.endswith("\n") else line + "\n")
                except Exception:
                    pass
            if keep:
                open(os.path.join(RES, f"runs_wimp{i}_{os.path.basename(fn)}"), "w").writelines(
                    keep
                )
        fd = os.path.join(d, "fields")
        if os.path.isdir(fd):
            for fn in os.listdir(fd):
                if fn[:-4] in need and not os.path.exists(os.path.join(RES, "fields", fn)):
                    shutil.copy(os.path.join(fd, fn), os.path.join(RES, "fields", fn))
        log("imported", d)
    done = {r["id"] for r in read_runs(RES) if "error" not in r}
    k1 = sum(
        1
        for arm in ("rbf", "mlp")
        for a in ALPHAS
        for s in S["seeds"][:5]
        if job_id(make_job(arm, 1, a, s)) in done
    )
    log(
        f"Stage-3 runs available: {len(done & need)}/{len(need)} | re-used k=1 runs (RBF + MLP, seeds 1-5): {k1}/20"
    )
    if k1 < 20:
        log(
            "NOTE: k=1 runs of the reproduction / Stage-1 datasets not found -> they will simply be trained here"
        )


def offline_checks():
    j = make_job("rbf", 1, 0.9, 1234)
    assert job_id(j) == "butterfly_gaussian_lhs_a0.9_N80_k1_scaled_ad5000_float32_s1234", job_id(j)
    j = make_job("mlp", 1, 0.1, 1234)
    assert (
        job_id(j) == "butterfly_mlp2x150_tanh_lhs_a0.1_N80_k1_scaled_ad5000_float32_s1234"
    ), job_id(j)
    ids = [job_id(j) for j in all_jobs()]
    assert len(ids) == len(set(ids)) == len(ARMS) * len(KS) * len(ALPHAS) * 10
    # exact solution / source consistency for every k (finite-difference check of the PDE operator)
    for k in KS:
        x, y, t, a, h = 0.31, -0.27, 0.6, 0.5, 1e-4
        lap = (
            exact_np(x + h, y, t, a, k)
            + exact_np(x - h, y, t, a, k)
            + exact_np(x, y + h, t, a, k)
            + exact_np(x, y - h, t, a, k)
            - 4 * exact_np(x, y, t, a, k)
        ) / h**2
        assert abs(lap + 2 * k * k * np.pi**2 * exact_np(x, y, t, a, k)) < 1e-3 * (1 + abs(lap))
    log(
        f"OFFLINE CHECKS PASSED | k = {KS}, alpha = {ALPHAS}, 10 seeds | RBF + MLP: {len(phase_jobs('core'))} runs"
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
    last = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        if time.time() - last > 1800:
            last = time.time()
            n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
            log(f"phase {phase}: {n}/{len(ids)} | {(time.time() - t0)/3600:.1f} h")
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
    if len(sm) != len(ARMS) or bad:
        raise SystemExit(f"SMOKE TEST FAILED -> nothing started: {bad or 'runs missing'}")
    log("SMOKE TEST PASSED (RBF and MLP)")
    left = lambda: S["hours"] - (time.time() - T0) / 3600
    if S["run"]:
        for ph in ["core"]:
            ids = {job_id(j) for j in phase_jobs(ph)}
            n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
            if n < len(ids) and left() > 0.4:
                run_phase(ph, left())
            n = len({r["id"] for r in read_runs(RES) if r["id"] in ids and "error" not in r})
            log(
                f"phase {ph}: {n}/{len(ids)}"
                + ("  -> COMPLETE" if n == len(ids) else "  -> continue in the next session")
            )
    try:
        reports()
    except Exception:
        traceback.print_exc()
        log("report generation failed -- results are still saved")
    shutil.make_archive("example42_stage3_results", "zip", ".", RES)
    log("DONE. figures/ and example42_stage3_results.zip are in the working directory")


if __name__ == "__main__":
    if sys.argv[1] == "all":
        main_all()
    else:
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
