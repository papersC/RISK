"""Experiment driver for the RISK-OCS evaluation.

Usage: python experiments.py <e1|hyst|e2|e3|e4|e5|e6|all>
Raw per-run summaries are written to results/<name>.pkl.
Seeds 200-209 are validation seeds (used only to match the hysteresis
threshold). Seeds 1000+ are test seeds.
"""
import sys, os, time, pickle
import numpy as np
from multiprocessing import Pool
from riskocs import runner, controllers as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
B_REF = 0.10      # exceedance threshold (equals the default budget)
HARM_TOL = 0.01   # harm tolerance (1% of an iteration)
VAL = list(range(200, 210))
TEST = list(range(1000, 1025))
DEFAULT_RISK = dict(budget=0.10, lam=0.5, alpha=0.9, S=16)


def summarise(fab, rec, iso):
    it = rec["it_post"]
    J, n = it.shape
    E = np.array(rec["cf_ell"]) if rec["cf_ell"] else np.zeros((0, J))
    ctrl = np.array(rec["ctrl_ms"]) if rec["ctrl_ms"] else np.zeros(1)
    return dict(
        name=rec["name"], fam=fab.wl.families,
        p50=np.percentile(it, 50, axis=1) / iso,
        p95=np.percentile(it, 95, axis=1) / iso,
        p99=np.percentile(it, 99, axis=1) / iso,
        mean=it.mean(1) / iso, it_norm=(it / iso[:, None]),
        thr=float((iso / it.mean(1)).sum()),
        switches=rec["switches"], decisions=rec["decisions"],
        harm=rec["harm"], exceed=rec["exceed"], jobiters=J * fab.cfg.NI,
        cf_ell=E, ctrl_ms=ctrl, opt_frac=rec["opt_mb"] / max(rec["opt_mb"] + rec["pkt_mb"], 1e-9),
        menu=np.mean(rec["menu"]) if rec["menu"] else 0.0)


def one(task):
    seed, ikw, cname, ckw, tagd = task
    fab = runner.instance(seed, **ikw)
    iso = np.array([runner.isolated_time(fab, j) for j in range(fab.wl.J)])
    ctrl = C.make(cname, **ckw)
    t0 = time.time()
    rec = fab.run(ctrl, b_ref=B_REF, harm_tol=HARM_TOL)
    w = fab.cfg.warmup
    rec["it_post"] = rec["itime"][:, w:]
    rec["name"] = ctrl.name
    out = summarise(fab, rec, iso)
    out.update(seed=seed, ikw=ikw, cname=cname, ckw=ckw, tag=tagd, wall=time.time() - t0)
    return out


def pool_run(tasks, name):
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(one, tasks, chunksize=1)
    with open(os.path.join(OUT, name + ".pkl"), "wb") as f:
        pickle.dump(res, f)
    print(f"{name}: {len(tasks)} runs in {time.time()-t0:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
def match_hysteresis():
    """Validation: pick theta so that nominal switching matches RISK-OCS."""
    grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    out = {}
    for mix in ("dense", "mixed", "moe"):
        ikw = dict(mix=mix, NI=30)
        tasks = [(s, ikw, "risk", DEFAULT_RISK, "risk") for s in VAL]
        tasks += [(s, ikw, "nominal", dict(theta=th), th) for s in VAL for th in grid]
        res = pool_run(tasks, f"val_{mix}")
        rate = lambda rs: np.mean([r["switches"] / r["jobiters"] for r in rs])
        target = rate([r for r in res if r["cname"] == "risk"])
        best = min(grid, key=lambda th: abs(rate([r for r in res if r["cname"] == "nominal" and r["tag"] == th]) - target))
        out[mix] = best
        print(mix, "target", target, "theta", best, flush=True)
    with open(os.path.join(OUT, "theta_match.pkl"), "wb") as f:
        pickle.dump(out, f)
    return out


def e1():
    th = pickle.load(open(os.path.join(OUT, "theta_match.pkl"), "rb"))
    tasks = []
    for mix in ("dense", "mixed", "moe"):
        ikw = dict(mix=mix, NI=50)
        ctrls = [("packet", {}), ("static", {}), ("reactive", {}), ("costaware", {}),
                 ("nominal", {}), ("nominal", dict(theta=th[mix])), ("risk", DEFAULT_RISK),
                 ("oracle", {})]
        if mix == "mixed":
            ctrls += [("risk_noguard", dict(lam=0.5)), ("risk_volonly", DEFAULT_RISK),
                      ("risk_mean", dict(budget=0.10)),
                      ("risk", dict(DEFAULT_RISK, guard="cvar"))]
        for s in TEST:
            for c, kw in ctrls:
                tasks.append((s, ikw, c, kw, mix))
    pool_run(tasks, "e1")


def e2():
    tasks = []
    for d in (2.0, 5.0, 10.0, 20.0, 40.0):
        for jit in (0.0, 0.15, 0.3, 0.5):
            ikw = dict(mix="mixed", NI=30, delta=d, jitter=jit)
            for s in TEST[:8]:
                for c, kw in (("nominal", {}), ("risk", DEFAULT_RISK), ("oracle", {})):
                    tasks.append((s, ikw, c, kw, (d, jit)))
    pool_run(tasks, "e2")


def e3():
    tasks = []
    ikw = dict(mix="mixed", NI=40)
    for s in TEST[:12]:
        for b in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
            tasks.append((s, ikw, "risk", dict(DEFAULT_RISK, budget=b), ("risk", b)))
        tasks.append((s, ikw, "risk_noguard", dict(lam=0.5), ("risk", np.inf)))
        for th in (0.0, 0.02, 0.05, 0.08, 0.12, 0.2):
            tasks.append((s, ikw, "nominal", dict(theta=th), ("hyst", th)))
    pool_run(tasks, "e3")


def e4():
    tasks = []
    for S in (4, 8, 16, 32, 64):
        tasks.append((TEST[0], dict(mix="mixed", NI=12), "risk", dict(DEFAULT_RISK, S=S), ("S", S)))
        tasks.append((TEST[1], dict(mix="mixed", NI=12), "risk", dict(DEFAULT_RISK, S=S), ("S", S)))
    for N in (16, 32, 64):
        J = N // 4
        mix = "mixed_%d" % J
        runner.MIXES[mix] = (runner.MIXES["mixed"] * (J // 4 + 1))[:J]
        for s in TEST[:2]:
            tasks.append((s, dict(mix=mix, N=N, NI=8 if N < 64 else 5), "risk", DEFAULT_RISK, ("N", N)))
            tasks.append((s, dict(mix=mix, N=N, NI=8 if N < 64 else 5), "nominal", {}, ("Nn", N)))
    with Pool(1) as p:  # one worker: clean timing
        res = p.map(one, tasks, chunksize=1)
    pickle.dump(res, open(os.path.join(OUT, "e4.pkl"), "wb"))
    print("e4 done", flush=True)


def e5():
    tasks = []
    ikw = dict(mix="mixed", NI=40)
    for s in TEST[:12]:
        tasks.append((s, ikw, "nominal", {}, ("nominal", -1, 0)))
        for pj in (0, 2):
            for b in (0.0, 0.05):
                tasks.append((s, ikw, "risk", dict(DEFAULT_RISK, budget=b, protected=[pj]), ("risk", pj, b)))
    pool_run(tasks, "e5")


def e6():
    tasks = []
    for name, ikw in (("partial", dict(partial=1)), ("ce0.5", dict(ce_ratio=0.5)),
                      ("ce1.0", dict(ce_ratio=1.0)), ("K4", dict(K=4)), ("H2", dict(Hmult=2.0)),
                      ("H8", dict(Hmult=8.0))):
        ikw = dict(ikw, mix="mixed", NI=30)
        for s in TEST[:8]:
            for c, kw in (("reactive", {}), ("nominal", {}), ("risk", DEFAULT_RISK), ("oracle", {})):
                tasks.append((s, ikw, c, kw, name))
    pool_run(tasks, "e6")


def e7():
    """Charge the measured median Python decision time (35 ms) instead of 1 ms."""
    tasks = []
    ikw = dict(mix="mixed", NI=30)
    for s in TEST[:8]:
        tasks.append((s, ikw, "risk", dict(DEFAULT_RISK, lat=35.0), "risk35"))
        tasks.append((s, ikw, "nominal", dict(lat=35.0), "nom35"))
    pool_run(tasks, "e7")


if __name__ == "__main__":
    what = sys.argv[1]
    steps = {"hyst": match_hysteresis, "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "e7": e7}
    if what == "all":
        for k in ("hyst", "e1", "e2", "e3", "e5", "e6", "e4"):
            steps[k]()
    else:
        steps[what]()
