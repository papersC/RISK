"""Python wrapper around the compiled core: state, cloning, replay, live run."""
import time
import numpy as np
from .sim_core import run_core, INF


class Config:
    def __init__(self, N=16, K=2, Co=50.0, ce_ratio=0.25, delta=10.0, partial=0,
                 H=40.0, NI=30, warmup=3):
        self.N, self.K, self.Co = N, K, Co
        self.Ce = Co * ce_ratio
        self.delta, self.partial = delta, partial
        self.H, self.NI, self.warmup = H, NI, warmup


class State:
    __slots__ = ("t", "it", "ph", "stg", "crem", "cstart", "frem", "istart", "done",
                 "X", "ready", "pendX", "pend_t", "itime", "iend", "stats")

    def clone(self):
        s = State.__new__(State)
        for k in State.__slots__:
            setattr(s, k, getattr(self, k).copy())
        return s


class Fabric:
    def __init__(self, cfg, wl, cm, vm, offsets, cont=1):
        self.cfg, self.wl, self.cm, self.vm = cfg, wl, cm, vm
        self.cont = cont  # 1: reactive continuation in replays, 0: hold continuation
        self.last_switch = -INF
        self.offsets = np.asarray(offsets, dtype=float)

    def init_state(self, X0):
        wl, cfg = self.wl, self.cfg
        J, F = wl.J, wl.Fmax
        s = State.__new__(State)
        s.t = np.zeros(1)
        s.it = np.zeros(J, dtype=np.int64)
        s.ph = np.zeros(J, dtype=np.int64)
        s.stg = np.zeros(J, dtype=np.int64)
        s.crem = np.array([self.offsets[j] + wl.cbase[j, 0] * self.cm[j, 0, 0] for j in range(J)])
        s.cstart = self.offsets.copy()
        s.frem = np.zeros((J, F))
        s.istart = self.offsets.copy()
        s.done = np.zeros(J, dtype=np.int64)
        s.X = X0.astype(np.int64).copy()
        s.ready = np.zeros((cfg.N, cfg.N))
        s.pendX = np.zeros((cfg.N, cfg.N), dtype=np.int64)
        s.pend_t = np.array([INF])
        s.itime = np.zeros((J, cfg.NI))
        s.iend = np.zeros(J)
        s.stats = np.zeros(3)
        return s

    CONT_LAT = 0.2  # decision latency charged to the continuation policy

    def _core(self, s, cm, vm, rowmode, mode, tag, stop_t=INF, cont=0, lastsw=None):
        wl, c = self.wl, self.cfg
        if lastsw is None:
            lastsw = np.array([-INF])
        return run_core(s.t, c.N, c.Co, c.Ce, c.delta, c.partial,
                        wl.nph, wl.nfl, wl.fsrc, wl.fdst, wl.fvol, wl.cbase,
                        cm, vm, rowmode,
                        s.it, s.ph, s.stg, s.crem, s.cstart, s.frem, s.istart, s.done,
                        s.X, s.ready, s.pendX, s.pend_t,
                        s.itime, s.iend, mode, stop_t, tag, c.NI, s.stats,
                        cont, c.K, c.H, self.CONT_LAT, lastsw)

    def replay(self, s0, Xa, lat, cm=None, vm=None, rowmode=1, hz=0):
        return self.replay_c(s0, Xa, lat, cm, vm, rowmode, self.cont, self.last_switch, hz)

    def replay_c(self, s0, Xa, lat, cm, vm, rowmode, cont, last_switch, hz=0):
        """Completion time of each job's current iteration, measured from now.
        Xa=None holds the incumbent. With rowmode=1 the (cm, vm) arrays are
        scenario rows applied to the current iteration; the elapsed part of
        an ongoing compute phase is preserved."""
        s = s0.clone()
        t0 = s.t[0]
        if rowmode == 1:
            wl = self.wl
            for j in range(wl.J):
                if s.done[j] == 0 and s.stg[j] == 0:
                    p = s.ph[j]
                    s.crem[j] = max(wl.cbase[j, p] * cm[j, 0, p] - (t0 - s.cstart[j]), 0.0)
        else:
            cm, vm = self.cm, self.vm
        lastsw = np.array([last_switch])
        if Xa is not None:
            s.pendX[:, :] = Xa
            s.pend_t[0] = t0 + lat
            lastsw[0] = t0 + lat
        tag = np.minimum(s.it + hz, self.cfg.NI - 1)
        code, _ = self._core(s, cm, vm, rowmode, 1, tag, cont=cont, lastsw=lastsw)
        if code < 0:
            raise RuntimeError("replay failed %d" % code)
        out = s.iend - t0
        out[s0.done == 1] = 0.0
        return out

    def run(self, ctrl, log_counterfactual=True, b_ref=0.05, harm_tol=0.01):
        cfg = self.cfg
        X0 = ctrl.initial_topology(self)
        s = self.init_state(X0)
        ctrl.reset(self, s)
        tag = -np.ones(self.wl.J, dtype=np.int64)
        last_switch = -INF
        self.last_switch = last_switch
        rec = dict(decisions=0, switches=0, harm=0, exceed=0, cf_gain=[], cf_worst=[], cf_ell=[],
                   ctrl_ms=[], menu=[])
        while True:
            code, trig = self._core(s, self.cm, self.vm, 0, 0, tag)
            if code == 0:
                break
            if code < 0:
                raise RuntimeError("core failure %d" % code)
            ctrl.observe(self, s)
            if ctrl.static:
                continue
            if s.pend_t[0] < INF:
                continue
            if s.t[0] < last_switch + cfg.H:
                continue
            t_a = time.perf_counter()
            Xn, info = ctrl.decide(self, s)
            rec["ctrl_ms"].append((time.perf_counter() - t_a) * 1e3)
            rec["decisions"] += 1
            if info is not None and "menu" in info:
                rec["menu"].append(info["menu"])
            if Xn is None or np.array_equal(Xn, s.X):
                continue
            rec["switches"] += 1
            if log_counterfactual:
                Th = self.replay(s, None, ctrl.lat, rowmode=0)
                Ta = self.replay(s, Xn, ctrl.lat, rowmode=0)
                act = s.done == 0
                ell = (Ta - Th)[act] / ctrl.Tbar[act]
                rec["cf_gain"].append(float(ell.mean()))
                rec["cf_worst"].append(float(ell.max()))
                full = np.full(self.wl.J, np.nan)
                full[act] = ell
                rec["cf_ell"].append(full)
                if ell.max() > harm_tol:
                    rec["harm"] += 1
                if ell.max() > b_ref:
                    rec["exceed"] += 1
            s.pendX[:, :] = Xn
            s.pend_t[0] = s.t[0] + ctrl.lat
            last_switch = s.t[0] + ctrl.lat
            self.last_switch = last_switch
        rec["itime"] = s.itime.copy()
        rec["opt_mb"], rec["pkt_mb"], rec["applied"] = s.stats.tolist()
        rec["makespan"] = float(s.t[0])
        return rec
