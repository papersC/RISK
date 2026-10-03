"""Build reproducible instances and run controllers on them."""
import numpy as np
from .fabric import Config, Fabric
from .workloads import Workload, realize, place_jobs
from . import controllers as C

MIXES = {
    "dense": ["dp", "pp", "dp", "pp"],
    "moe": ["moe", "moe", "dp", "moe"],
    "mixed": ["dp", "pp", "moe", "moe"],
}


def instance(seed, mix="mixed", N=16, K=2, m=8, jitter=0.3, delta=10.0, H=None,
             ce_ratio=0.25, NI=30, partial=0, Hmult=4.0, vol_cv=None,
             migrate_scale=1.0, stagger=100.0):
    rng = np.random.default_rng(seed)
    fams = MIXES[mix]
    J = len(fams)
    plc = place_jobs(N, J, m, rng)
    wl = Workload(fams, plc, N)
    cm, vm = realize(wl, NI, jitter, rng, vol_cv=vol_cv, migrate_scale=migrate_scale)
    offsets = rng.uniform(0, stagger, J)
    cfg = Config(N=N, K=K, ce_ratio=ce_ratio, delta=delta,
                 H=(Hmult * delta if H is None else H), NI=NI, partial=partial)
    # calibration trace from an independent stream (no test information)
    crng = np.random.default_rng(seed + 10_000_019)
    ccm, cvm = realize(wl, C.W_HIST, jitter, crng, vol_cv=vol_cv, migrate_scale=migrate_scale)
    est = np.array([wl.cbase[j, :wl.nph[j]].sum() * 1.5 for j in range(J)])
    cal = (ccm, cvm, [[e] for e in est])
    # calibration iteration times from a static-topology run on the calibration trace
    cfab = Fabric(Config(N=N, K=K, ce_ratio=ce_ratio, delta=delta, H=cfg.H,
                         NI=C.W_HIST, partial=partial), wl, ccm, cvm, offsets)
    cfab.calibration, cfab.ctrl_seed = cal, seed
    rec = cfab.run(C.Static(), log_counterfactual=False)
    cT = [list(rec["itime"][j, 1:]) for j in range(J)]
    fab = Fabric(cfg, wl, cm, vm, offsets)
    fab.calibration = (ccm, cvm, cT)
    fab.ctrl_seed = seed + 7
    return fab


def run(fab, ctrl):
    rec = fab.run(ctrl)
    w = fab.cfg.warmup
    it = rec["itime"][:, w:]
    rec["it_post"] = it
    rec["name"] = ctrl.name
    return rec


def isolated_time(fab, j):
    """Iteration time of job j alone, no jitter, own static matching."""
    from .workloads import Workload
    wl = fab.wl
    one = Workload([wl.families[j]], [wl.placements[j]], fab.cfg.N)
    cm = np.ones((1, 6, one.Pmax))
    vm = np.ones((1, 6, one.Pmax, one.Fmax))
    cfg = Config(N=fab.cfg.N, K=fab.cfg.K, ce_ratio=fab.cfg.Ce / fab.cfg.Co,
                 delta=fab.cfg.delta, H=fab.cfg.H, NI=6)
    f1 = Fabric(cfg, one, cm, vm, [0.0])
    f1.calibration = (cm[:, :5], vm[:, :5], [[1.0]])
    f1.ctrl_seed = 0
    rec = f1.run(C.Static(), log_counterfactual=False)
    return float(np.median(rec["itime"][0, 1:]))
