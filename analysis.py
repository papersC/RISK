"""Statistics, tables and figures for the RISK-OCS paper.

Reads results/*.pkl and writes figures to figures/ and LaTeX number
macros to figures/numbers.tex. Confidence intervals use a percentile
bootstrap over seeds (whole runs are the resampling unit).
"""
import os, pickle, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(12345)
B = 4000

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
    "pdf.fonttype": 42, "lines.linewidth": 1.2})
COL = {"Packet only": "#9AA5B1", "Static OCS": "#6B7785", "Reactive": "#C8811A",
       "Cost-aware": "#8C6BB1", "Nominal": "#1F3A5F", "Hysteresis": "#4C9BE8",
       "RISK-OCS": "#B5443B", "Clairvoyant": "#1B7F79"}
MARK = {"Packet only": "v", "Static OCS": "s", "Reactive": "^", "Cost-aware": "D",
        "Nominal": "o", "Hysteresis": "P", "RISK-OCS": "*", "Clairvoyant": "X"}


def load(name):
    return pickle.load(open(os.path.join(RES, name + ".pkl"), "rb"))


def label(r):
    if r["cname"] == "nominal" and r["ckw"].get("theta", 0) > 0:
        return "Hysteresis"
    if r["cname"] == "risk" and r["ckw"].get("guard") == "cvar":
        return "CVaR guard"
    return r["name"]


def metrics(r):
    return dict(p95=float(np.mean(r["p95"])), worst=float(np.max(r["p95"])),
                p99=float(np.mean(r["p99"])), mean=float(np.mean(r["mean"])),
                thr=r["thr"] / len(r["p95"]),
                sw=100.0 * r["switches"] / r["jobiters"],
                harm_share=r["harm"] / max(r["switches"], 1),
                harm100=100.0 * r["harm"] / r["jobiters"],
                exc100=100.0 * r["exceed"] / r["jobiters"],
                opt=r["opt_frac"], ctrl_p50=float(np.percentile(r["ctrl_ms"], 50)),
                ctrl_p99=float(np.percentile(r["ctrl_ms"], 99)))


def boot_mean(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(a, b):
    """Mean relative difference (a-b)/b over seeds with a bootstrap CI."""
    a, b = np.asarray(a), np.asarray(b)
    d = (a - b) / b
    return boot_mean(d)


def by(res, key):
    out = collections.defaultdict(dict)
    for r in res:
        out[key(r)][r["seed"]] = r
    return out


NUM = {}


def num(k, v, fmt="{:.1f}"):
    NUM[k] = fmt.format(v)


# ---------------------------------------------------------------------------
def table_e1():
    res = load("e1")
    lines = []
    order = ["Packet only", "Static OCS", "Reactive", "Cost-aware", "Nominal", "Hysteresis",
             "RISK-OCS", "Clairvoyant"]
    ab = ["No job guard", "No timing scenarios", "Guard + mean only", "CVaR guard"]
    summary = {}
    for mix in ("dense", "mixed", "moe"):
        rr = [r for r in res if r["tag"] == mix]
        g = by(rr, label)
        summary[mix] = {}
        for name, d in g.items():
            seeds = sorted(d)
            M = [metrics(d[s]) for s in seeds]
            summary[mix][name] = dict(seeds=seeds, M=M)
    pickle.dump(summary, open(os.path.join(RES, "e1_summary.pkl"), "wb"))
    # LaTeX table of main results
    tex = []
    for mix, title in (("dense", "Dense"), ("mixed", "Mixed"), ("moe", "MoE heavy")):
        tex.append(r"\multicolumn{8}{@{}l}{\textit{%s tenant mix}}\\" % title)
        for name in order + (ab if mix == "mixed" else []):
            if name not in summary[mix]:
                continue
            M = summary[mix][name]["M"]
            f = lambda k: np.mean([m[k] for m in M])
            p, lo, hi = boot_mean([m["p95"] for m in M])
            hs = (np.sum([m["harm_share"] * m["sw"] for m in M]) / max(np.sum([m["sw"] for m in M]), 1e-9))
            tex.append(f"{name} & {p:.2f} [{lo:.2f}, {hi:.2f}] & {f('worst'):.2f} & {f('thr'):.3f} & "
                       f"{f('sw'):.1f} & {100*hs:.0f} & {f('exc100'):.1f} & {100*f('opt'):.0f}\\\\")
        tex.append(r"\midrule")
    tex = tex[:-1]
    open(os.path.join(FIG, "table_main.tex"), "w").write("\n".join(tex) + "\n")
    # key numbers (mixed)
    S = summary
    for mix in ("dense", "mixed", "moe"):
        nom = S[mix]["Nominal"]; rk = S[mix]["RISK-OCS"]; hy = S[mix]["Hysteresis"]; cl = S[mix]["Clairvoyant"]
        rea = S[mix]["Reactive"]; st = S[mix]["Static OCS"]
        a = lambda X, k: [m[k] for m in X["M"]]
        m_, lo, hi = paired(a(rk, "p95"), a(nom, "p95")); num(f"{mix}RiskVsNomPnf", 100*m_); num(f"{mix}RiskVsNomPnfLo", 100*lo); num(f"{mix}RiskVsNomPnfHi", 100*hi)
        m_, lo, hi = paired(a(rk, "p95"), a(hy, "p95")); num(f"{mix}RiskVsHyPnf", 100*m_); num(f"{mix}RiskVsHyPnfLo", 100*lo); num(f"{mix}RiskVsHyPnfHi", 100*hi)
        m_, lo, hi = paired(a(cl, "p95"), a(nom, "p95")); num(f"{mix}ClVsNomPnf", 100*m_); num(f"{mix}ClVsNomPnfLo", 100*lo); num(f"{mix}ClVsNomPnfHi", 100*hi)
        m_, lo, hi = paired(a(nom, "p95"), a(st, "p95")); num(f"{mix}NomVsStPnf", 100*m_)
        m_, lo, hi = paired(a(nom, "p95"), a(rea, "p95")); num(f"{mix}NomVsReaPnf", 100*m_)
        m_, lo, hi = paired(a(rk, "thr"), a(nom, "thr")); num(f"{mix}RiskVsNomThr", 100*m_)
        m_, lo, hi = paired(a(rk, "thr"), a(hy, "thr")); num(f"{mix}RiskVsHyThr", 100*m_)
        for nm, X in (("Nom", nom), ("Risk", rk), ("Hy", hy), ("Rea", rea), ("Cl", cl)):
            sw = np.sum(a(X, "sw"))
            hs = np.sum(np.array(a(X, "harm_share")) * np.array(a(X, "sw"))) / max(sw, 1e-9)
            num(f"{mix}{nm}HarmShare", 100*hs, "{:.0f}")
            num(f"{mix}{nm}Exc", np.mean(a(X, "exc100")))
            num(f"{mix}{nm}Sw", np.mean(a(X, "sw")))
            num(f"{mix}{nm}Pnf", np.mean(a(X, "p95")), "{:.2f}")
        e_r, e_h = np.mean(a(rk, "exc100")), np.mean(a(hy, "exc100"))
        num(f"{mix}ExcCutVsHy", 100*(1-e_r/max(e_h, 1e-9)), "{:.0f}")
        e_n = np.mean(a(nom, "exc100"))
        num(f"{mix}ExcCutVsNom", 100*(1-e_r/max(e_n, 1e-9)), "{:.0f}")
    return summary


def fig_main(summary):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45))
    order = ["Static OCS", "Reactive", "Cost-aware", "Nominal", "Hysteresis",
             "RISK-OCS", "Clairvoyant"]
    for ax, (mix, title) in zip(axs, (("dense", "(a) Dense mix"), ("mixed", "(b) Mixed"), ("moe", "(c) MoE heavy"))):
        for name in order:
            M = summary[mix][name]["M"]
            x = boot_mean([m["thr"] for m in M]); y = boot_mean([m["p95"] for m in M])
            ax.errorbar(x[0], y[0], xerr=[[x[0]-x[1]], [x[2]-x[0]]], yerr=[[y[0]-y[1]], [y[2]-y[0]]],
                        fmt=MARK[name], color=COL[name], ms=6 if name == "RISK-OCS" else 4.5,
                        mec="white", mew=0.4, elinewidth=0.7, capsize=1.5, label=name, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Normalised throughput")
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Mean per-job p95 slowdown")
    for ax in axs:
        ax.annotate("better", xy=(0.97, 0.05), xytext=(0.80, 0.22), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6), fontsize=6.5, color="0.4")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False,
               handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "fig_main.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_harm(summary):
    res = load("e1")
    rr = [r for r in res if r["tag"] == "mixed"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    for name in ("Reactive", "Nominal", "Hysteresis", "RISK-OCS", "Clairvoyant"):
        w = np.concatenate([np.nanmax(r["cf_ell"], axis=1) for r in rr if label(r) == name and len(r["cf_ell"])])
        x = np.sort(w); y = np.arange(1, len(x)+1)/len(x)
        ax.plot(x, y, color=COL[name], label=name, lw=1.5 if name == "RISK-OCS" else 1.1)
        if name in ("Nominal", "RISK-OCS", "Hysteresis"):
            num("mixed%sMedWorst" % name.replace("-", "").replace(" ", ""), 100*np.median(w), "{:.1f}")
            num("mixed%sQninetyWorst" % name.replace("-", "").replace(" ", ""), 100*np.quantile(w, 0.9), "{:.0f}")
    ax.axvline(0.10, color="k", ls=":", lw=0.8); ax.text(0.105, 0.08, "$b=0.10$", fontsize=7)
    ax.axvline(0.01, color="gray", ls=":", lw=0.6)
    ax.set_xlim(-0.3, 0.6)
    ax.set_xlabel("Realised worst-job loss of a committed switch\n(fraction of an iteration, vs. holding)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(a) Who pays for a reconfiguration", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.4)
    ax = axs[1]
    names = ["Reactive", "Cost-aware", "Nominal", "Hysteresis", "RISK-OCS", "Clairvoyant"]
    S = summary["mixed"]
    hs = [100*np.sum([m["harm_share"]*m["sw"] for m in S[n]["M"]])/np.sum([m["sw"] for m in S[n]["M"]]) for n in names]
    ex = [np.mean([m["exc100"] for m in S[n]["M"]]) for n in names]
    xx = np.arange(len(names))
    ax.bar(xx-0.2, hs, 0.4, color=[COL[n] for n in names], alpha=0.9, label="Harmful share of switches (%)")
    ax2 = ax.twinx()
    ax2.bar(xx+0.2, ex, 0.4, color="white", edgecolor=[COL[n] for n in names], hatch="////", lw=0.8,
            label="Exceedances per 100 job-iterations")
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(xx); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Harmful share of switches (%)  [solid]")
    ax2.set_ylabel("Exceedances / 100 job-iter.  [hatched]")
    ax.set_title("(b) Harm and budget exceedance", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_harm.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_regime():
    res = load("e2")
    deltas = sorted({r["tag"][0] for r in res}); jits = sorted({r["tag"][1] for r in res})
    P = np.zeros((len(jits), len(deltas))); E = np.zeros_like(P); O = np.zeros_like(P)
    for i, jt in enumerate(jits):
        for k, d in enumerate(deltas):
            g = by([r for r in res if r["tag"] == (d, jt)], label)
            seeds = sorted(g["Nominal"])
            pn = [np.mean(g["Nominal"][s]["p95"]) for s in seeds]
            pr = [np.mean(g["RISK-OCS"][s]["p95"]) for s in seeds]
            po = [np.mean(g["Clairvoyant"][s]["p95"]) for s in seeds]
            P[i, k] = 100*np.mean((np.array(pr)-pn)/pn)
            O[i, k] = 100*np.mean((np.array(po)-pn)/pn)
            en = np.mean([g["Nominal"][s]["exceed"]/g["Nominal"][s]["jobiters"] for s in seeds])
            er = np.mean([g["RISK-OCS"][s]["exceed"]/g["RISK-OCS"][s]["jobiters"] for s in seeds])
            E[i, k] = 100*(1-er/en) if en > 0 else np.nan
    pickle.dump(dict(P=P, E=E, O=O, deltas=deltas, jits=jits), open(os.path.join(RES, "e2_summary.pkl"), "wb"))
    num("regimePmin", P.min()); num("regimePmax", P.max())
    num("regimeEmin", np.nanmin(E), "{:.0f}"); num("regimeEmax", np.nanmax(E), "{:.0f}")
    num("regimeOmin", O.min()); num("regimeOmax", O.max())
    k40 = deltas.index(40.0); num("regimePdFortyMean", P[:, k40].mean())
    k2 = deltas.index(2.0); num("regimePdTwoMean", P[:, k2].mean())
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25))
    for ax, M, t, cmap, lim in ((axs[0], P, "(a) RISK-OCS p95 vs. nominal (%)", "RdBu_r", max(abs(P.min()), abs(P.max()))),
                                 (axs[1], E, "(b) Exceedance reduction (%)", "Greens", None),
                                 (axs[2], O, "(c) Clairvoyant p95 vs. nominal (%)", "RdBu_r", max(abs(O.min()), abs(O.max()), 1))):
        if lim is None:
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, origin="lower", aspect="auto")
        else:
            im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                v = M[i, k]
                ax.text(k, i, "–" if np.isnan(v) else f"{v:+.0f}" if lim else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if (lim and abs(v) > 0.6*lim) or (not lim and v > 65) else "black")
        ax.set_xticks(range(len(deltas))); ax.set_xticklabels([f"{d:g}" for d in deltas])
        ax.set_yticks(range(len(jits))); ax.set_yticklabels([f"{j:g}" for j in jits])
        ax.set_xlabel(r"Switching delay $\delta$ (ms)")
        ax.set_title(t, loc="left")
        for s in ax.spines.values(): s.set_visible(False)
    axs[0].set_ylabel("Compute jitter (CV)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_regime.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_frontier():
    res = load("e3")
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for kind, color, mk, lab in (("risk", COL["RISK-OCS"], "*", "RISK-OCS (budget $b$)"),
                                 ("hyst", COL["Hysteresis"], "P", r"Nominal + hysteresis ($\theta$)")):
        vals = sorted({r["tag"][1] for r in res if r["tag"][0] == kind})
        X, Y, L = [], [], []
        for v in vals:
            rr = [r for r in res if r["tag"] == (kind, v)]
            X.append(np.mean([100*r["exceed"]/r["jobiters"] for r in rr]))
            Y.append(np.mean([np.mean(r["p95"]) for r in rr]))
            L.append(v)
        ax.plot(X, Y, "-", color=color, marker=mk, ms=6 if mk == "*" else 4.5, label=lab, mec="white", mew=0.4)
        for q, (x, y, v) in enumerate(zip(X, Y, L)):
            t = "∞" if np.isinf(v) else f"{v:g}"
            off = (-12, 2) if q % 2 == 0 else (-4, -9)
            if kind == "hyst":
                off = (3, 3) if q % 2 == 0 else (2, -9)
            ax.annotate(t, (x, y), textcoords="offset points", xytext=off, fontsize=6, color=color)
        NUM_KEY = "front" + kind
        pickle.dump(dict(X=X, Y=Y, L=L), open(os.path.join(RES, f"e3_{kind}.pkl"), "wb"))
    ax.set_xlabel("Budget exceedances per 100 job-iterations")
    ax.set_ylabel("Mean per-job p95 slowdown")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_frontier.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_protect():
    res = load("e5")
    nom = {r["seed"]: r for r in res if r["tag"][0] == "nominal"}
    rows = []
    for pj, pname in ((0, "DP tenant"), (2, "MoE tenant")):
        for b in (None, 0.05, 0.0):
            if b is None:
                rr = list(nom.values()); lab = "Nominal"
            else:
                rr = [r for r in res if r["tag"] == ("risk", pj, b)]; lab = f"Protected, $b={b:g}$"
            exc = np.mean([100*np.nansum(r["cf_ell"][:, pj] > 0.05)/r["jobiters"] if len(r["cf_ell"]) else 0 for r in rr])
            p95 = np.mean([r["p95"][pj] for r in rr]); mean = np.mean([r["mean"][pj] for r in rr])
            others = np.mean([np.mean(np.delete(r["p95"], pj)) for r in rr])
            rows.append((pname, lab, exc, p95, mean, others))
    pickle.dump(rows, open(os.path.join(RES, "e5_summary.pkl"), "wb"))
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.2))
    cols = ["#1F3A5F", "#E3A3A0", "#B5443B"]
    for g, pname in enumerate(("DP tenant", "MoE tenant")):
        sub = [r for r in rows if r[0] == pname]
        for i, r in enumerate(sub):
            axs[0].bar(g*4+i, r[2], color=cols[i], label=r[1] if g == 0 else None)
            axs[1].bar(g*4+i, r[3], color=cols[i])
            axs[1].plot([g*4+i-0.35, g*4+i+0.35], [r[4]]*2, color="k", lw=1.0)
    for ax in axs:
        ax.set_xticks([1, 5]); ax.set_xticklabels(["Protected DP tenant", "Protected MoE tenant"])
        ax.grid(axis="y", alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Local exceedances of $b=0.05$\nper 100 job-iterations")
    axs[0].set_title("(a) Per-decision protection", loc="left")
    axs[1].set_ylabel("Protected tenant slowdown\n(bar: p95, line: mean)")
    axs[1].set_title("(b) End-to-end outcome", loc="left")
    axs[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_protect.pdf"), bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        k = ("DP" if r[0].startswith("DP") else "MoE") + {"Nominal": "Nom", "Protected, $b=0.05$": "Bfive", "Protected, $b=0$": "Bzero"}[r[1]]
        num("prot%sExc" % k, r[2]); num("prot%sPnf" % k, r[3], "{:.2f}"); num("prot%sMean" % k, r[4], "{:.2f}"); num("prot%sOthers" % k, r[5], "{:.2f}")


def fig_overhead():
    res = load("e4")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.1))
    S = sorted({r["tag"][1] for r in res if r["tag"][0] == "S"})
    for q, ls in ((50, "-"), (99, "--")):
        axs[0].plot(S, [np.percentile(np.concatenate([r["ctrl_ms"] for r in res if r["tag"] == ("S", s)]), q) for s in S],
                    ls, marker="o", color=COL["RISK-OCS"], ms=3.5, label=f"p{q}")
    axs[0].axhline(40, color="k", ls=":", lw=0.8); axs[0].text(S[0], 44, "holding interval $H$ = 40 ms", fontsize=6.5)
    axs[0].set_xscale("log", base=2); axs[0].set_yscale("log")
    axs[0].set_xlabel("Scenarios $S$ (N = 16)"); axs[0].set_ylabel("Decision time (ms)")
    axs[0].set_title("(a) Scenario count", loc="left"); axs[0].legend(frameon=False)
    Ns = sorted({r["tag"][1] for r in res if r["tag"][0] == "N"})
    for tag, col, nm in (("N", COL["RISK-OCS"], "RISK-OCS"), ("Nn", COL["Nominal"], "Nominal")):
        for q, ls in ((50, "-"), (99, "--")):
            axs[1].plot(Ns, [np.percentile(np.concatenate([r["ctrl_ms"] for r in res if r["tag"] == (tag, n)]), q) for n in Ns],
                        ls, marker="o", ms=3.5, color=col, label=f"{nm} p{q}")
    axs[1].axhline(40, color="k", ls=":", lw=0.8)
    axs[1].set_xscale("log", base=2); axs[1].set_yscale("log")
    axs[1].set_xlabel("Racks $N$ (tenants $J=N/4$, $S=16$)"); axs[1].set_ylabel("Decision time (ms)")
    axs[1].set_title("(b) Fabric size", loc="left"); axs[1].legend(frameon=False, ncol=2, fontsize=6)
    for ax in axs:
        ax.grid(alpha=0.25, lw=0.4, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_overhead.pdf"), bbox_inches="tight")
    plt.close(fig)
    for s in S:
        c = np.concatenate([r["ctrl_ms"] for r in res if r["tag"] == ("S", s)])
        num(f"ovS{s}p50", np.percentile(c, 50)); num(f"ovS{s}p99", np.percentile(c, 99))
    for n in Ns:
        c = np.concatenate([r["ctrl_ms"] for r in res if r["tag"] == ("N", n)])
        num(f"ovN{n}p50", np.percentile(c, 50)); num(f"ovN{n}p99", np.percentile(c, 99))
        c = np.concatenate([r["ctrl_ms"] for r in res if r["tag"] == ("Nn", n)])
        num(f"ovNn{n}p50", np.percentile(c, 50)); num(f"ovNn{n}p99", np.percentile(c, 99))


def table_e6():
    res = load("e6")
    names = ["default", "partial", "ce0.5", "ce1.0", "K4", "H2", "H8"]
    desc = {"default": "Default configuration", "partial": "Regional outage (unchanged circuits serve)", "ce0.5": r"$C_e/C_o=0.5$",
            "ce1.0": r"$C_e/C_o=1$", "K4": r"$K=4$ ports per rack", "H2": r"$H=2\delta$", "H8": r"$H=8\delta$"}
    tex = []
    for n in names:
        g = by([r for r in res if r["tag"] == n], label)
        seeds = sorted(g["Nominal"])
        row = [desc[n]]
        for c in ("Reactive", "Nominal", "RISK-OCS", "Clairvoyant"):
            row.append(f"{np.mean([np.mean(g[c][s]['p95']) for s in seeds]):.2f}")
        en = np.mean([100*g['Nominal'][s]['exceed']/g['Nominal'][s]['jobiters'] for s in seeds])
        er = np.mean([100*g['RISK-OCS'][s]['exceed']/g['RISK-OCS'][s]['jobiters'] for s in seeds])
        row.append(f"{en:.1f} / {er:.1f}")
        tex.append(" & ".join(row) + r"\\")
    open(os.path.join(FIG, "table_sens.tex"), "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    import sys
    have = lambda n: os.path.exists(os.path.join(RES, n + ".pkl"))
    if have("e1"):
        s = table_e1(); fig_main(s); fig_harm(s)
    if have("e2"): fig_regime()
    if have("e3"): fig_frontier()
    if have("e5"): fig_protect()
    if have("e4"): fig_overhead()
    if have("e6"): table_e6()
    with open(os.path.join(FIG, "numbers.tex"), "w") as f:
        for k, v in sorted(NUM.items()):
            f.write("\\newcommand{\\n%s}{%s}\n" % (k.replace(".", "").replace("0", "Z").replace("1", "O").replace("2", "T").replace("3", "Th").replace("4", "F").replace("5", "Fi").replace("6", "Si").replace("8", "E").replace("9", "N"), v))
    print("analysis done", len(NUM), "numbers")
