"""Event-driven fluid simulator for a hybrid optical/packet fabric.

Units: time in milliseconds, volume in megabytes (MB), rate in MB/ms.
One MB/ms equals 8 Gb/s, so 400 Gb/s is 50 MB/ms.

The same compiled core serves the live run (mode 0) and the
candidate replays used by the admission gate (mode 1). Using a single
implementation guarantees that the controller evaluates candidates with
exactly the service policy that later executes them.
"""
import numpy as np
from numba import njit

INF = 1e18
EPS = 1e-9


@njit(cache=True)
def _allocate(N, Co, Ce, t, J, Fmax, stg, nfl_cur, fsrc_cur, fdst_cur, frem,
              X, ready, rate, isopt, npair, eg_rem, in_rem, frozen):
    """Fill rate[j,f]. Circuit flows share their circuit equally.
    Remaining flows receive max-min fair packet rates subject to per-rack
    egress and ingress capacity Ce (progressive filling)."""
    # reset counters
    for i in range(N):
        eg_rem[i] = Ce
        in_rem[i] = Ce
    for j in range(J):
        if stg[j] != 1:
            continue
        for f in range(nfl_cur[j]):
            s = fsrc_cur[j, f]
            d = fdst_cur[j, f]
            npair[s, d] = 0
    # optical membership
    for j in range(J):
        if stg[j] != 1:
            continue
        for f in range(nfl_cur[j]):
            rate[j, f] = 0.0
            isopt[j, f] = 0
            frozen[j, f] = 1
            if frem[j, f] <= 0.0:
                continue
            s = fsrc_cur[j, f]
            d = fdst_cur[j, f]
            if X[s, d] == 1 and ready[s, d] <= t + EPS:
                isopt[j, f] = 1
                npair[s, d] += 1
            else:
                frozen[j, f] = 0
    for j in range(J):
        if stg[j] != 1:
            continue
        for f in range(nfl_cur[j]):
            if isopt[j, f] == 1:
                s = fsrc_cur[j, f]
                d = fdst_cur[j, f]
                rate[j, f] = Co / npair[s, d]
    # progressive filling on packet fabric
    eg_cnt = np.zeros(N, dtype=np.int64)
    in_cnt = np.zeros(N, dtype=np.int64)
    for _ in range(4 * N + 4):
        for i in range(N):
            eg_cnt[i] = 0
            in_cnt[i] = 0
        any_unfrozen = False
        for j in range(J):
            if stg[j] != 1:
                continue
            for f in range(nfl_cur[j]):
                if frozen[j, f] == 0:
                    any_unfrozen = True
                    eg_cnt[fsrc_cur[j, f]] += 1
                    in_cnt[fdst_cur[j, f]] += 1
        if not any_unfrozen:
            break
        inc = INF
        for i in range(N):
            if eg_cnt[i] > 0:
                v = eg_rem[i] / eg_cnt[i]
                if v < inc:
                    inc = v
            if in_cnt[i] > 0:
                v = in_rem[i] / in_cnt[i]
                if v < inc:
                    inc = v
        if inc < 0.0:
            inc = 0.0
        for j in range(J):
            if stg[j] != 1:
                continue
            for f in range(nfl_cur[j]):
                if frozen[j, f] == 0:
                    rate[j, f] += inc
                    eg_rem[fsrc_cur[j, f]] -= inc
                    in_rem[fdst_cur[j, f]] -= inc
        tol = 1e-9 * Ce
        for j in range(J):
            if stg[j] != 1:
                continue
            for f in range(nfl_cur[j]):
                if frozen[j, f] == 0:
                    if eg_rem[fsrc_cur[j, f]] <= tol or in_rem[fdst_cur[j, f]] <= tol:
                        frozen[j, f] = 1


@njit(cache=True)
def greedy_bmatch(W, K, Xinc, out):
    """Greedy degree-K b-matching on weights W (largest first, index ties).
    Ports left free keep incumbent circuits. Result written into out."""
    N = W.shape[0]
    for a in range(N):
        for b in range(N):
            out[a, b] = 0
    n = 0
    for a in range(N):
        for b in range(N):
            if a != b and W[a, b] > 0.0:
                n += 1
    if n == 0:
        for a in range(N):
            for b in range(N):
                out[a, b] = Xinc[a, b]
        return
    key = np.empty(n)
    ia = np.empty(n, dtype=np.int64)
    ib = np.empty(n, dtype=np.int64)
    k = 0
    for a in range(N):
        for b in range(N):
            if a != b and W[a, b] > 0.0:
                key[k] = -W[a, b]
                ia[k] = a
                ib[k] = b
                k += 1
    order = np.argsort(key, kind="mergesort")
    od = np.zeros(N, dtype=np.int64)
    idg = np.zeros(N, dtype=np.int64)
    for q in range(n):
        a = ia[order[q]]
        b = ib[order[q]]
        if od[a] < K and idg[b] < K:
            out[a, b] = 1
            od[a] += 1
            idg[b] += 1
    for a in range(N):
        for b in range(N):
            if Xinc[a, b] == 1 and out[a, b] == 0 and od[a] < K and idg[b] < K:
                out[a, b] = 1
                od[a] += 1
                idg[b] += 1


@njit(cache=True)
def run_core(tarr, N, Co, Ce, delta, partial,
             nph, nfl, fsrc, fdst, fvol, cbase,
             cm, vm, rowmode,
             it, ph, stg, crem, cstart, frem, istart, done,
             X, ready, pendX, pend_t,
             itime, iend, mode, stop_t, tag, NI, stats,
             cont, K, H, lat_c, lastsw):
    """Advance the fabric state.

    mode 0: return code 1 when any job releases a communication phase
            (a decision trigger), 0 when all jobs finished NI iterations,
            2 when stop_t is reached.
    mode 1: replay; return 0 once every job j has completed iteration tag[j].
    rowmode: if 1, compute and volume multipliers are read from row 0
            (scenario arrays); otherwise from the iteration index.
    cont: if 1 (replay only), the common continuation policy is applied at
            every communication release: the reactive greedy matching of
            queued bytes is installed when it differs from the incumbent,
            subject to the holding interval H and decision latency lat_c.
    stats: [optical MB, packet MB, switches applied]
    Returns trigger job index (or -1).
    """
    J = nph.shape[0]
    Fmax = fsrc.shape[2]
    rate = np.zeros((J, Fmax))
    isopt = np.zeros((J, Fmax), dtype=np.int64)
    frozen = np.zeros((J, Fmax), dtype=np.int64)
    npair = np.zeros((N, N), dtype=np.int64)
    eg_rem = np.zeros(N)
    in_rem = np.zeros(N)
    nfl_cur = np.zeros(J, dtype=np.int64)
    fsrc_cur = np.zeros((J, Fmax), dtype=np.int64)
    fdst_cur = np.zeros((J, Fmax), dtype=np.int64)
    for j in range(J):
        if stg[j] == 1:
            p = ph[j]
            nfl_cur[j] = nfl[j, p]
            for f in range(nfl[j, p]):
                fsrc_cur[j, f] = fsrc[j, p, f]
                fdst_cur[j, f] = fdst[j, p, f]
    t = tarr[0]
    guard = 0
    while True:
        guard += 1
        if guard > 2000000:
            tarr[0] = t
            return -2, -1
        # replay termination
        if mode == 1:
            fin = True
            for j in range(J):
                if done[j] == 0 and it[j] <= tag[j]:
                    fin = False
                    break
            if fin:
                tarr[0] = t
                return 0, -1
        else:
            alld = True
            for j in range(J):
                if done[j] == 0:
                    alld = False
                    break
            if alld:
                tarr[0] = t
                return 0, -1
        # apply pending switch
        if pend_t[0] <= t + EPS:
            for a in range(N):
                for b in range(N):
                    if pendX[a, b] == 1:
                        if partial == 1 and X[a, b] == 1:
                            pass  # unchanged circuit keeps serving
                        else:
                            ready[a, b] = t + delta
                    X[a, b] = pendX[a, b]
            pend_t[0] = INF
            stats[2] += 1.0
        _allocate(N, Co, Ce, t, J, Fmax, stg, nfl_cur, fsrc_cur, fdst_cur, frem,
                  X, ready, rate, isopt, npair, eg_rem, in_rem, frozen)
        # next event
        dt = INF
        for j in range(J):
            if done[j] == 1:
                continue
            if stg[j] == 0:
                if crem[j] < dt:
                    dt = crem[j]
            else:
                for f in range(nfl_cur[j]):
                    if frem[j, f] > 0.0 and rate[j, f] > 0.0:
                        v = frem[j, f] / rate[j, f]
                        if v < dt:
                            dt = v
        for a in range(N):
            for b in range(N):
                if X[a, b] == 1 and ready[a, b] > t + EPS:
                    v = ready[a, b] - t
                    if v < dt:
                        dt = v
        if pend_t[0] < INF:
            v = pend_t[0] - t
            if v < dt:
                dt = v
        if mode == 0 and stop_t - t < dt:
            dt = stop_t - t
        if dt < 0.0:
            dt = 0.0
        if dt >= INF:
            tarr[0] = t
            return -3, -1
        # advance
        t += dt
        for j in range(J):
            if done[j] == 1:
                continue
            if stg[j] == 0:
                crem[j] -= dt
            else:
                for f in range(nfl_cur[j]):
                    if frem[j, f] > 0.0:
                        moved = rate[j, f] * dt
                        if isopt[j, f] == 1:
                            stats[0] += min(moved, frem[j, f])
                        else:
                            stats[1] += min(moved, frem[j, f])
                        frem[j, f] -= moved
                        if frem[j, f] < 1e-7:
                            frem[j, f] = 0.0
        # process transitions
        trig = -1
        for j in range(J):
            if done[j] == 1:
                continue
            changed = True
            while changed and done[j] == 0:
                changed = False
                if stg[j] == 0 and crem[j] <= 1e-9:
                    # compute finished: release communication
                    p = ph[j]
                    stg[j] = 1
                    row = 0 if rowmode == 1 else it[j]
                    nfl_cur[j] = nfl[j, p]
                    for f in range(nfl[j, p]):
                        fsrc_cur[j, f] = fsrc[j, p, f]
                        fdst_cur[j, f] = fdst[j, p, f]
                        frem[j, f] = fvol[j, p, f] * vm[j, row, p, f]
                    trig = j
                    changed = True
                elif stg[j] == 1:
                    alldone = True
                    for f in range(nfl_cur[j]):
                        if frem[j, f] > 0.0:
                            alldone = False
                            break
                    if alldone:
                        ph[j] += 1
                        if ph[j] >= nph[j]:
                            itime[j, it[j]] = t - istart[j]
                            if it[j] == tag[j]:
                                iend[j] = t
                            it[j] += 1
                            istart[j] = t
                            ph[j] = 0
                            if it[j] >= NI:
                                done[j] = 1
                                stg[j] = 0
                                nfl_cur[j] = 0
                                break
                        stg[j] = 0
                        nfl_cur[j] = 0
                        row = 0 if rowmode == 1 else it[j]
                        crem[j] = cbase[j, ph[j]] * cm[j, row, ph[j]]
                        cstart[j] = t
                        changed = True
        if mode == 0 and trig >= 0:
            tarr[0] = t
            return 1, trig
        if mode == 1 and cont == 1 and trig >= 0 and pend_t[0] >= INF and t >= lastsw[0] + H - EPS:
            Wc = np.zeros((N, N))
            for j in range(J):
                if stg[j] == 1:
                    for f in range(nfl_cur[j]):
                        if frem[j, f] > 0.0:
                            Wc[fsrc_cur[j, f], fdst_cur[j, f]] += frem[j, f]
            greedy_bmatch(Wc, K, X, pendX)
            same = True
            for a in range(N):
                for b in range(N):
                    if pendX[a, b] != X[a, b]:
                        same = False
            if not same:
                pend_t[0] = t + lat_c
                lastsw[0] = t + lat_c
        if mode == 0 and t >= stop_t - EPS:
            tarr[0] = t
            return 2, -1
