"""Topology controllers: baselines, nominal replay gate and RISK-OCS."""
import numpy as np
from .sim_core import greedy_bmatch

W_HIST = 20  # sliding history window (iterations)


def kmatch(W, K, Xinc=None):
    """Greedy degree-K b-matching (same routine as the replay continuation);
    unused ports keep incumbent circuits."""
    N = W.shape[0]
    if Xinc is None:
        Xinc = np.zeros((N, N), dtype=np.int64)
    out = np.zeros((N, N), dtype=np.int64)
    greedy_bmatch(np.ascontiguousarray(W, dtype=np.float64), K,
                  np.ascontiguousarray(Xinc, dtype=np.int64), out)
    return out


def cvar(L, alpha):
    """Empirical CVaR of equally weighted losses (Rockafellar and Uryasev)."""
    L = np.sort(np.asarray(L))[::-1]
    S = len(L)
    if S == 1:
        return float(L[0])
    tail = (1.0 - alpha) * S
    acc, tot = 0.0, 0.0
    for x in L:
        w = min(1.0, tail - tot)
        if w <= 0:
            break
        acc += w * x
        tot += w
    return acc / tot


class Base:
    static = False
    lat = 0.2

    def __init__(self, name):
        self.name = name

    # ----- history and forecasts -------------------------------------------------
    def reset(self, fab, s, cal=None):
        wl = fab.wl
        self.rng = np.random.default_rng(fab.ctrl_seed)
        ccm, cvm, cT = fab.calibration
        self.hcm = [list(ccm[j]) for j in range(wl.J)]
        self.hvm = [list(cvm[j]) for j in range(wl.J)]
        self.hT = [list(cT[j]) for j in range(wl.J)]
        self.nobs = np.zeros(wl.J, dtype=int)
        self._refresh(wl)

    def observe(self, fab, s):
        wl = fab.wl
        changed = False
        for j in range(wl.J):
            while self.nobs[j] < s.it[j]:
                i = self.nobs[j]
                self.hcm[j].append(fab.cm[j, i].copy())
                self.hvm[j].append(fab.vm[j, i].copy())
                if i > 0:
                    self.hT[j].append(s.itime[j, i])
                self.nobs[j] += 1
                changed = True
        if changed:
            self._refresh(wl)

    def _refresh(self, wl):
        J = wl.J
        self.cm_pred = np.ones((J, 1, wl.Pmax))
        self.vm_pred = np.ones((J, 1, wl.Pmax, wl.Fmax))
        self.cm_q90 = np.ones((J, 1, wl.Pmax))
        self.Tbar = np.zeros(J)
        for j in range(J):
            H = np.array(self.hcm[j][-W_HIST:])
            self.cm_pred[j, 0] = np.median(H, axis=0)
            self.cm_q90[j, 0] = np.quantile(H, 0.9, axis=0)
            V = self.hvm[j][-W_HIST:]
            e = V[0].copy()
            for v in V[1:]:
                e = 0.5 * e + 0.5 * v
            self.vm_pred[j, 0] = e
            self.Tbar[j] = float(np.mean(self.hT[j][-W_HIST:]))

    # ----- demand view used for candidate construction -------------------------
    def demand(self, fab, s, horizon):
        """Return (Wbytes, Wready, per-job bytes matrices) over a short window."""
        wl, cfg = fab.wl, fab.cfg
        N = cfg.N
        t = s.t[0]
        Wb = np.zeros((N, N))
        Wr = np.zeros((N, N))
        per = np.zeros((wl.J, N, N))
        for j in range(wl.J):
            if s.done[j]:
                continue
            p = s.ph[j]
            if s.stg[j] == 1:
                tau = 0.0
                vols = s.frem[j, :wl.nfl[j, p]]
            else:
                tau = max(wl.cbase[j, p] * self.cm_pred[j, 0, p] - (t - s.cstart[j]), 0.0)
                vols = wl.fvol[j, p, :wl.nfl[j, p]] * self.vm_pred[j, 0, p, :wl.nfl[j, p]]
            if tau > horizon:
                continue
            src = wl.fsrc[j, p, :wl.nfl[j, p]]
            dst = wl.fdst[j, p, :wl.nfl[j, p]]
            np.add.at(per[j], (src, dst), vols)
            Wr += per[j] / (1.0 + tau / cfg.delta)
        Wb = per.sum(0)
        return Wb, Wr, per

    def menu(self, fab, s):
        cfg = fab.cfg
        horizon = cfg.H + cfg.delta + self.lat
        Wb, Wr, per = self.demand(fab, s, horizon)
        cands = [s.X.copy()]
        # most delayed job relative to its typical iteration time
        prog = np.where(s.done == 0, (s.t[0] - s.istart) / np.maximum(self.Tbar, 1e-6), -1)
        jd = int(np.argmax(prog))
        for W in (Wb, Wb + 9.0 * per[jd], Wr):
            if W.max() <= 0:
                continue
            Xc = kmatch(W, cfg.K, s.X)
            if not any(np.array_equal(Xc, c) for c in cands):
                cands.append(Xc)
        # incremental candidate: keep circuits that carry active traffic and
        # assign only the remaining ports
        busy = (per.sum(0) > 0) & (s.X == 1)
        if Wb.max() > 0:
            W2 = Wb.copy() + busy * 1e12
            Xc = kmatch(W2, cfg.K, s.X)
            if not any(np.array_equal(Xc, c) for c in cands):
                cands.append(Xc)
        return cands

    def initial_topology(self, fab):
        return np.zeros((fab.cfg.N, fab.cfg.N), dtype=np.int64)


class PacketOnly(Base):
    static = True

    def __init__(self):
        super().__init__("Packet only")


class Static(Base):
    static = True

    def __init__(self):
        super().__init__("Static OCS")

    def initial_topology(self, fab):
        wl = fab.wl
        ccm, cvm, _ = fab.calibration
        W = np.zeros((fab.cfg.N, fab.cfg.N))
        for j in range(wl.J):
            vbar = np.mean(cvm[j], axis=0)
            for p in range(wl.nph[j]):
                n = wl.nfl[j, p]
                np.add.at(W, (wl.fsrc[j, p, :n], wl.fdst[j, p, :n]), wl.fvol[j, p, :n] * vbar[p, :n])
        return kmatch(W, fab.cfg.K)


class Reactive(Base):
    """Switch to the matching of currently queued bytes whenever it differs."""

    def __init__(self):
        super().__init__("Reactive")

    def decide(self, fab, s):
        Wb, _, _ = self.demand(fab, s, 0.0)
        if Wb.max() <= 0:
            return None, None
        return kmatch(Wb, fab.cfg.K, s.X), None


class CostAware(Base):
    """Eclipse-inspired amortisation: pick the menu entry that maximises
    predicted bytes served in the window net of the switching outage."""

    def __init__(self):
        super().__init__("Cost-aware")

    def decide(self, fab, s):
        cfg = fab.cfg
        Wwin = cfg.H + cfg.delta
        Wb, _, _ = self.demand(fab, s, Wwin)
        cands = self.menu(fab, s)
        best, bi = -1.0, 0
        for i, Xc in enumerate(cands):
            dur = Wwin - (cfg.delta if i > 0 else 0.0)
            served = np.minimum(Wb, cfg.Co * dur) * Xc
            gain = served.sum()
            if gain > best + 1e-9:
                best, bi = gain, i
        if bi == 0:
            return None, {"menu": len(cands)}
        return cands[bi], {"menu": len(cands)}


class ReplayGate(Base):
    """Incumbent-relative admission by causal replay.

    scen: 'point'  single point forecast (nominal controller)
          'boot'   block-bootstrap joint scenarios plus two stressed cases
          'volonly' bootstrap of volumes only, compute at point forecast
          'oracle' realised future (clairvoyant one-step reference)
    """
    lat = 1.0

    def __init__(self, name, scen="boot", S=16, lam=0.5, alpha=0.9, budget=np.inf,
                 theta=0.0, guard="max", protected=None, hz=0):
        super().__init__(name)
        self.hz = hz
        self.guard = guard
        self.protected = protected  # None: every job is protected
        self.scen, self.S, self.lam, self.alpha = scen, S, lam, alpha
        self.budget, self.theta = budget, theta

    def scenarios(self, fab):
        wl = fab.wl
        if self.scen == "point":
            return [(self.cm_pred, self.vm_pred)]
        if self.scen == "oracle":
            return [(None, None)]
        out = []
        nb = self.S - 2 if self.scen == "boot" else self.S
        for _ in range(nb):
            cm = np.ones_like(self.cm_pred)
            vm = np.ones_like(self.vm_pred)
            for j in range(wl.J):
                Hc = self.hcm[j][-W_HIST:]
                Hv = self.hvm[j][-W_HIST:]
                b = self.rng.integers(len(Hc))
                cm[j, 0] = Hc[b] if self.scen == "boot" else self.cm_pred[j, 0]
                vm[j, 0] = Hv[b]
            out.append((cm, vm))
        if self.scen == "boot":
            out.append((self.cm_q90, self.vm_pred))  # stressed late release
            vm = self.vm_pred.copy()  # most migrated observed hotspot pattern
            for j in range(wl.J):
                Hv = self.hvm[j][-W_HIST:]
                d = [np.abs(v - self.vm_pred[j, 0]).sum() for v in Hv]
                vm[j, 0] = Hv[int(np.argmax(d))]
            out.append((self.cm_pred, vm))
        return out

    def decide(self, fab, s):
        cands = self.menu(fab, s)
        if len(cands) == 1:
            return None, {"menu": 1}
        scen = self.scenarios(fab)
        act = s.done == 0
        Tb = np.maximum(self.Tbar, 1e-6)
        T0 = []
        for cm, vm in scen:
            if cm is None:
                T0.append(fab.replay(s, None, self.lat, rowmode=0, hz=self.hz))
            else:
                T0.append(fab.replay(s, None, self.lat, cm, vm, 1, hz=self.hz))
        best_i, best_F = 0, 0.0
        for i in range(1, len(cands)):
            ell = np.zeros((len(scen), act.sum()))
            for k, (cm, vm) in enumerate(scen):
                if cm is None:
                    Ta = fab.replay(s, cands[i], self.lat, rowmode=0, hz=self.hz)
                else:
                    Ta = fab.replay(s, cands[i], self.lat, cm, vm, 1, hz=self.hz)
                ell[k] = ((Ta - T0[k]) / Tb)[act]
            if np.isfinite(self.budget):
                act_idx = np.nonzero(act)[0]
                cols = [q for q, j in enumerate(act_idx)
                        if self.protected is None or j in self.protected]
                if not cols:
                    worst = -np.inf
                elif self.guard == "max":
                    worst = ell[:, cols].max()
                else:  # per-job CVaR of the slowdown
                    worst = max(cvar(ell[:, q], self.alpha) for q in cols)
                if worst > self.budget:
                    continue
            L = ell.mean(axis=1)
            F = (1 - self.lam) * L.mean() + self.lam * cvar(L, self.alpha)
            if F < best_F:
                best_F, best_i = F, i
        info = {"menu": len(cands), "F": best_F}
        if best_i == 0 or best_F >= -self.theta:
            return None, info
        return cands[best_i], info


def make(name, **kw):
    lat = kw.pop("lat", None)
    c = _make(name, **kw)
    if lat is not None:
        c.lat = lat
    return c


def _make(name, **kw):
    if name == "packet":
        return PacketOnly()
    if name == "static":
        return Static()
    if name == "reactive":
        return Reactive()
    if name == "costaware":
        return CostAware()
    if name == "nominal":
        return ReplayGate("Nominal", scen="point", S=1, lam=0.0, **kw)
    if name == "oracle":
        return ReplayGate("Clairvoyant", scen="oracle", S=1, lam=0.0, **kw)
    if name == "risk":
        kw.setdefault("budget", 0.05)
        return ReplayGate("RISK-OCS", scen="boot", **kw)
    if name == "risk_noguard":
        return ReplayGate("No job guard", scen="boot", budget=np.inf, **kw)
    if name == "risk_volonly":
        kw.setdefault("budget", 0.05)
        return ReplayGate("No timing scenarios", scen="volonly", **kw)
    if name == "risk_mean":
        kw.setdefault("budget", 0.05)
        return ReplayGate("Guard + mean only", scen="boot", lam=0.0, **kw)
    raise ValueError(name)
