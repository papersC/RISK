"""Synthetic training workloads with explicit, disclosed parameters.

Three families are modelled as chains of (compute, communication) phases:
  dp   : data-parallel ring AllReduce of a gradient bucket
  pp   : two pipeline replicas of four stages each with forward and
         backward activation transfers and a replica gradient exchange
  moe  : expert dispatch and combine with skewed, drifting expert load
         followed by a dense-parameter ring AllReduce

The generator is synthetic. Published model sizes are not used to claim
production realism; the parameters below are the full specification.
"""
import numpy as np

# Base parameters (ms, MB). All values are disclosed in the paper.
FAMILY = {
    "dp": dict(compute=[80.0], grad_mb=1500.0),
    "pp": dict(compute=[30.0, 45.0, 6.0], act_mb=400.0, grad_mb=800.0),
    "moe": dict(compute=[20.0, 25.0, 20.0], token_mb=1400.0, dense_mb=300.0,
                hot_sigma=0.9, drift_phi=0.85, migrate_p=0.08),
}


def ring_flows(racks, vol_per_rack):
    m = len(racks)
    per = 2.0 * (m - 1) / m * vol_per_rack
    return [(racks[i], racks[(i + 1) % m], per) for i in range(m)]


def build_job(family, racks):
    """Return list of phases. Each phase = (compute_ms, [(src,dst,MB),...], kind)."""
    p = FAMILY[family]
    m = len(racks)
    phases = []
    if family == "dp":
        phases.append((p["compute"][0], ring_flows(racks, p["grad_mb"]), "ring"))
    elif family == "pp":
        half = m // 2
        reps = [racks[:half], racks[half:]]
        fwd, bwd, dpx = [], [], []
        for r in reps:
            for s in range(half - 1):
                fwd.append((r[s], r[s + 1], p["act_mb"]))
                bwd.append((r[s + 1], r[s], p["act_mb"]))
        for s in range(half):
            dpx.append((reps[0][s], reps[1][s], p["grad_mb"]))
            dpx.append((reps[1][s], reps[0][s], p["grad_mb"]))
        phases.append((p["compute"][0], fwd, "fwd"))
        phases.append((p["compute"][1], bwd, "bwd"))
        phases.append((p["compute"][2], dpx, "dpx"))
    elif family == "moe":
        per = p["token_mb"] / (m - 1)
        disp = [(racks[a], racks[b], per) for a in range(m) for b in range(m) if a != b]
        comb = [(d, s, v) for (s, d, v) in disp]
        phases.append((p["compute"][0], disp, "dispatch"))
        phases.append((p["compute"][1], comb, "combine"))
        phases.append((p["compute"][2], ring_flows(racks, p["dense_mb"]), "ring"))
    else:
        raise ValueError(family)
    return phases


class Workload:
    """Packs job templates into dense arrays for the compiled core."""

    def __init__(self, families, placements, N):
        self.families = list(families)
        self.placements = [list(r) for r in placements]
        self.N = N
        self.jobs = [build_job(f, r) for f, r in zip(families, placements)]
        J = len(self.jobs)
        Pmax = max(len(ph) for ph in self.jobs)
        Fmax = max(len(fl) for ph in self.jobs for (_, fl, _) in ph)
        self.J, self.Pmax, self.Fmax = J, Pmax, Fmax
        self.nph = np.zeros(J, dtype=np.int64)
        self.nfl = np.zeros((J, Pmax), dtype=np.int64)
        self.fsrc = np.zeros((J, Pmax, Fmax), dtype=np.int64)
        self.fdst = np.zeros((J, Pmax, Fmax), dtype=np.int64)
        self.fvol = np.zeros((J, Pmax, Fmax))
        self.cbase = np.zeros((J, Pmax))
        self.kind = [[k for (_, _, k) in ph] for ph in self.jobs]
        for j, ph in enumerate(self.jobs):
            self.nph[j] = len(ph)
            for p, (c, fl, _) in enumerate(ph):
                self.cbase[j, p] = c
                self.nfl[j, p] = len(fl)
                for f, (s, d, v) in enumerate(fl):
                    self.fsrc[j, p, f] = s
                    self.fdst[j, p, f] = d
                    self.fvol[j, p, f] = v


def realize(wl, NI, jitter, rng, straggler_p=0.05, phi=0.6, rho=0.7,
            vol_cv=None, migrate_scale=1.0):
    """Draw realised compute and volume multipliers.

    Compute multipliers are log-normal with coefficient of variation close to
    `jitter`, correlated within an iteration (rho) and across iterations
    (AR(1) coefficient phi), with occasional heavy stragglers.
    MoE volume multipliers follow drifting expert popularity with abrupt
    hotspot migrations. Dense-family volumes are deterministic.
    """
    J, P, F = wl.J, wl.Pmax, wl.Fmax
    cm = np.ones((J, NI, P))
    vm = np.ones((J, NI, P, F))
    sig = np.sqrt(np.log(1.0 + jitter ** 2)) if jitter > 0 else 0.0
    for j in range(J):
        g = rng.standard_normal()
        for i in range(NI):
            g = phi * g + np.sqrt(1 - phi ** 2) * rng.standard_normal()
            strag = 1.0
            if jitter > 0 and rng.random() < straggler_p:
                strag = 1.0 + rng.uniform(0.3, 1.0) * min(1.0, jitter / 0.3)
            for p in range(wl.nph[j]):
                z = rho * g + np.sqrt(1 - rho ** 2) * rng.standard_normal()
                cm[j, i, p] = np.exp(sig * z - 0.5 * sig ** 2) * strag
        if wl.families[j] == "moe":
            prm = FAMILY["moe"]
            racks = wl.placements[j]
            m = len(racks)
            hs = prm["hot_sigma"] if vol_cv is None else vol_cv
            th = rng.standard_normal(m) * hs
            pos = {r: k for k, r in enumerate(racks)}
            for i in range(NI):
                th = prm["drift_phi"] * th + np.sqrt(1 - prm["drift_phi"] ** 2) * rng.standard_normal(m) * hs
                if rng.random() < prm["migrate_p"] * migrate_scale:
                    th = rng.permutation(th)
                w = np.exp(th)
                for p in range(wl.nph[j]):
                    kind = wl.kind[j][p]
                    if kind not in ("dispatch", "combine"):
                        continue
                    for f in range(wl.nfl[j, p]):
                        s = wl.fsrc[j, p, f]
                        d = wl.fdst[j, p, f]
                        expert = pos[d] if kind == "dispatch" else pos[s]
                        src = pos[s] if kind == "dispatch" else pos[d]
                        tot = w.sum() - w[src]
                        vm[j, i, p, f] = (m - 1) * w[expert] / tot
    return cm, vm


def place_jobs(N, J, m, rng, overlap=True):
    """Random rack placement. With overlap, racks host several tenants."""
    if not overlap:
        perm = rng.permutation(N)
        return [sorted(perm[k * m:(k + 1) * m].tolist()) for k in range(J)]
    out = []
    load = np.zeros(N)
    for _ in range(J):
        # prefer lightly loaded racks to keep tenancy balanced
        key = load + rng.random(N) * 0.9
        pick = np.argsort(key)[:m]
        load[pick] += 1
        out.append(rng.permutation(pick).tolist())
    return out
