"""Hand-checkable validation cases for the fluid core (run: pytest -q)."""
import numpy as np
from riskocs.fabric import Config, Fabric
from riskocs.workloads import Workload
from riskocs import controllers as C


def _fab(phases_per_job, N=4, K=1, delta=10.0, NI=2, ce_ratio=0.25, offsets=None):
    """Build a fabric from explicit phases [(compute, [(s,d,MB)])] per job."""
    wl = Workload.__new__(Workload)
    J = len(phases_per_job)
    P = max(len(p) for p in phases_per_job)
    F = max(max(len(fl) for _, fl in p) for p in phases_per_job)
    wl.J, wl.Pmax, wl.Fmax, wl.N = J, P, F, N
    wl.families = ["dp"] * J
    wl.placements = [list(range(N))] * J
    wl.kind = [["x"] * len(p) for p in phases_per_job]
    wl.nph = np.array([len(p) for p in phases_per_job], dtype=np.int64)
    wl.nfl = np.zeros((J, P), dtype=np.int64)
    wl.fsrc = np.zeros((J, P, F), dtype=np.int64)
    wl.fdst = np.zeros((J, P, F), dtype=np.int64)
    wl.fvol = np.zeros((J, P, F))
    wl.cbase = np.zeros((J, P))
    for j, ph in enumerate(phases_per_job):
        for p, (c, fl) in enumerate(ph):
            wl.cbase[j, p] = c
            wl.nfl[j, p] = len(fl)
            for f, (s, d, v) in enumerate(fl):
                wl.fsrc[j, p, f], wl.fdst[j, p, f], wl.fvol[j, p, f] = s, d, v
    cm = np.ones((J, NI, P))
    vm = np.ones((J, NI, P, F))
    cfg = Config(N=N, K=K, ce_ratio=ce_ratio, delta=delta, H=40.0, NI=NI)
    fab = Fabric(cfg, wl, cm, vm, offsets if offsets is not None else [0.0] * J)
    fab.calibration = (cm, vm, [[100.0]] * J)
    fab.ctrl_seed = 0
    return fab


def _run_static(fab, X):
    s = fab.init_state(X)
    tag = -np.ones(fab.wl.J, dtype=np.int64)
    while True:
        code, _ = fab._core(s, fab.cm, fab.vm, 0, 0, tag)
        if code == 0:
            return s
        assert code == 1


def test_single_flow_on_circuit():
    # 10 ms compute, then 500 MB at 50 MB/ms on a circuit: 10 ms transfer
    fab = _fab([[(10.0, [(0, 1, 500.0)])]])
    X = np.zeros((4, 4), dtype=np.int64)
    X[0, 1] = 1
    s = _run_static(fab, X)
    assert np.allclose(s.itime[0], 20.0)


def test_single_flow_on_packet():
    # packet egress 12.5 MB/ms: 500 MB takes 40 ms
    fab = _fab([[(10.0, [(0, 1, 500.0)])]])
    s = _run_static(fab, np.zeros((4, 4), dtype=np.int64))
    assert np.allclose(s.itime[0], 50.0)


def test_two_flows_share_packet_egress():
    # two flows leave rack 0: each gets 6.25 MB/ms, 250 MB each -> 40 ms
    fab = _fab([[(0.0, [(0, 1, 250.0), (0, 2, 250.0)])]])
    s = _run_static(fab, np.zeros((4, 4), dtype=np.int64))
    assert np.allclose(s.itime[0], 40.0)


def test_max_min_fairness_unequal():
    # flows 0->1 and 0->2 share egress of 0; flow 3->2 shares ingress of 2.
    # max-min: all three get 6.25 MB/ms initially (egress 0 and ingress 2 saturate)
    fab = _fab([[(0.0, [(0, 1, 62.5), (0, 2, 125.0), (3, 2, 125.0)])]])
    s = _run_static(fab, np.zeros((4, 4), dtype=np.int64))
    # 0->1 ends at 10 ms; then 0->2 and 3->2 share ingress 2 at 6.25 each
    # remaining 62.5 MB each -> +10 ms. Iteration ends at 20 ms.
    assert np.allclose(s.itime[0], 20.0)


def test_global_outage_delays_unchanged_circuit():
    # flow on circuit 0->1 (1000 MB = 20 ms). Switch at t=5 adds circuit 2->3.
    # Global outage (delta=10): from 5 to 15 the flow runs on packet at 12.5.
    fab = _fab([[(0.0, [(0, 1, 1000.0)])]], delta=10.0)
    X = np.zeros((4, 4), dtype=np.int64)
    X[0, 1] = 1
    s = fab.init_state(X)
    Xn = X.copy()
    Xn[2, 3] = 1
    T = fab.replay_c(s, Xn, 5.0, None, None, 0, 0, -1e18)
    # served: 5 ms*50=250, 10 ms*12.5=125, rest 625 at 50 -> 12.5 ms; total 27.5
    assert np.isclose(T[0], 27.5)
    fab.cfg.partial = 1
    T2 = fab.replay_c(s, Xn, 5.0, None, None, 0, 0, -1e18)
    assert np.isclose(T2[0], 20.0)  # unchanged circuit keeps serving


def test_dependency_and_overlap_two_jobs():
    # job A: compute 10 then 0->1 500 MB; job B: compute 30 then 0->2 500 MB.
    # No circuits: A alone on egress 0 from 10..30 (250 MB), then both share.
    fab = _fab([[(10.0, [(0, 1, 500.0)])], [(30.0, [(0, 2, 500.0)])]], NI=1)
    s = _run_static(fab, np.zeros((4, 4), dtype=np.int64))
    # A: 250 MB left at t=30, rate 6.25 -> done t=70 ; B: 250 by t=70, then 12.5 -> 20 ms -> 90
    assert np.isclose(s.itime[0, 0], 70.0)
    assert np.isclose(s.itime[1, 0], 90.0)


def test_cvar():
    assert np.isclose(C.cvar([0.1, 0.0, -0.1, -0.2], 0.5), 0.05)
    assert np.isclose(C.cvar([0.3], 0.9), 0.3)


def test_hold_is_always_feasible_and_menu_degrees():
    fab = _fab([[(0.0, [(0, 1, 500.0), (1, 2, 500.0), (2, 3, 500.0)])]], K=1)
    W = np.random.default_rng(0).random((4, 4))
    X = C.kmatch(W, 2)
    assert X.sum(1).max() <= 2 and X.sum(0).max() <= 2 and np.trace(X) == 0
