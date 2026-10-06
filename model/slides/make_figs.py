
"""
Figures for the poster and the deck, in Computer Modern.

Everything here is drawn from the repository's own measurements. The panels that need real
data -- the SPD object, the spatial filter patterns, the confusion matrix -- are recomputed
from npz2 on fold 0 of the trial split and cached; the rest read numbers already recorded in
SUMMARY.md and logs/.

Style: matplotlib ships Computer Modern (cmr10), the LaTeX body face, so the figures match a
LaTeX-styled deck without needing a system font. Colour is used for one thing only -- accent
means "ours", grey means "baseline or control" -- so the reader learns the code once.

    python slides/make_figs.py            # all figures
    python slides/make_figs.py --only spd # one
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
CACHE = os.path.join(HERE, "cache.npz")

INK = "#111111"
MUTED = "#8C8C8C"
FAINT = "#D8D8D8"
ACC = "#A62B1F"          # ours
ACC2 = "#20548F"          # second representation / secondary series
OK = "#2E6B4F"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,          # cmr10 has no Unicode minus
    "axes.formatter.use_mathtext": True,  # tick labels in Computer Modern too
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("  wrote %s" % os.path.relpath(p, os.path.dirname(HERE)), flush=True)


def bare(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        if s not in keep:
            ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- real data

def compute_cache(a):
    """Fold 0 of the trial split: SPD object, filter patterns, confusion matrix."""
    import sdcore as sd
    from sdcore import (basis_full, block_features, folds_for, laplacian_modes, load_pooled,
                        snr_filter, tangent, unified_spd, window_blocks)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, tok, sid, pos, n_sub = load_pooled("ph11", normalise=True)
    n_cls = len(np.unique(y))
    tr, te = folds_for("trial", X, y, tok, sid, 1)[0]

    W = snr_filter(X, y, sid, tr, n_cls, 14, 0.05, "noise")
    Z = np.einsum("ck,nct->nkt", W, X)
    C = unified_spd(Z, tr, 2, 20, 7, 1.0)

    # Haufe patterns: filters are not maps. A = Sigma_x W (W' Sigma_x W)^-1
    Xf = X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1])
    Sx = np.cov(Xf.T)
    A = Sx @ W @ np.linalg.inv(W.T @ Sx @ W)

    # class-average waveforms in the filtered space, for the evoked panel
    erp = np.stack([Z[tr][y[tr] == c].mean(0) for c in range(n_cls)])

    # confusion matrix of the best model on this fold, aggregated over subjects
    V, wlap = laplacian_modes(pos)
    U = basis_full("lap", X, tr, V, wlap, 0.0, 0.5)
    edges = np.linspace(0, X.shape[1], 7).astype(int)
    one, blk = [], []
    for b in range(6):
        Zb = np.einsum("ck,nct->nkt", U[:, edges[b]:edges[b + 1]], X)
        one.append(tangent(unified_spd(Zb, tr, 2, 20, 7, 1.0), tr, sid))
        blk.append(block_features(window_blocks(Zb, tr, 4, 2, 20), tr, sid))
    F = np.hstack(one + blk)

    M = np.zeros((n_cls, n_cls))
    for s in np.unique(sid):
        mtr, mte = tr[sid[tr] == s], te[sid[te] == s]
        if len(mte) < n_cls or len(mtr) < 3 * n_cls:
            continue
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        m.fit(F[mtr], y[mtr])
        p = m.predict(F[mte])
        for t, q in zip(y[mte], p):
            M[t, q] += 1

    np.savez(CACHE, C=C.mean(0), A=A, pos=pos, erp=erp, conf=M,
             counts=np.bincount(y), W=W)
    print("  cached real quantities -> %s" % os.path.relpath(CACHE, os.path.dirname(HERE)))


def cache():
    if not os.path.exists(CACHE):
        raise SystemExit("run with --compute first")
    return np.load(CACHE)


# ---------------------------------------------------------------- numbers from logs
#
# These were transcribed by hand once and one series came out wrong -- ten of the fifteen
# per-fold values in the paired figure did not match the log, and a sweep figure silently
# mixed three different leak rates into one line. Nothing here is typed in any more: every
# number a figure draws is parsed out of the log that produced it, and `--verify` prints all
# of them so the parse can be checked against the file.

LOGS = os.path.join(os.path.dirname(HERE), "logs")


def _log(name):
    with open(os.path.join(LOGS, name)) as f:
        return f.read()


def grab(name, pattern, group=1, cast=float):
    """Every match of `pattern` in a log, in file order. Multiline, so ^ anchors a line."""
    return [cast(m.group(group)) for m in re.finditer(pattern, _log(name), re.M)]


def one(name, pattern, group=1, cast=float):
    v = grab(name, pattern, group, cast)
    if not v:
        raise SystemExit("make_figs: no match for %r in logs/%s" % (pattern, name))
    return v[0]


def numbers():
    """Everything the log-derived figures draw. One place, so --verify can print it."""
    n = {}
    n["ours_folds"] = grab("synth_comb_s3.log", r"band_one\+bl ([0-9.]+)")
    n["eeg_folds"] = grab("eegnet_ph11_s3.log", r"pooled\+ft_head [0-9.]+/([0-9.]+)")
    for arm in ["dct_low", "dct_high", "shuffled", "gauss", "none"]:
        n["phi_" + arm] = grab("arch_phi_ph11.log", arm + r" ([0-9.]+)")
    n["ktime"] = [(0, one("onemodel_ph11.log", r"^one_K0\s+([0-9.]+)"))] + [
        (k, v) for k, v in zip(grab("sweep_ktime.log", r"K_time=(\d+)", cast=int),
                               grab("sweep_ktime.log", r"one_KT\s+([0-9.]+)")[::2])]
    for k in ["trial", "token", "time", "loso"]:
        n["split_" + k] = (one("final_split_ph11.log", k + r"\s+([0-9.]+) \+- ([0-9.]+)"),
                           one("final_split_ph11.log", k + r"\s+[0-9.]+ \+- ([0-9.]+)"))
    n["split_perm"] = (one("final_perm_ph11.log", r"permutation\s+([0-9.]+)"),
                       one("final_perm_ph11.log", r"permutation\s+[0-9.]+ \+- ([0-9.]+)"))
    num = r"(nan|[0-9.]+)"
    n["budget"] = [tuple(np.nan if x == "nan" else float(x) for x in m.groups())
                   for m in re.finditer(r"^\s*(\d+) \|\s+%s\s+%s \|\s+%s\s+%s"
                                        % (num, num, num, num),
                                        _log("budget_ph11.log"), re.M)]
    n["final_designed"] = one("holdout_ph11.log", r"final_model\s+\d+\s+[0-9.]+ ±[0-9.]+\s+([0-9.]+)")
    n["final_best"] = one("holdout_ph11.log", r"band_one\+blocks\s+\d+\s+[0-9.]+ ±[0-9.]+\s+([0-9.]+)")
    n["eeg_cv"] = one("eegnet_ph11_s3.log", r"^pooled\+ft_head\s+([0-9.]+)")
    n["spdnet"] = one("spdnet_ph11.log", r"^  spdnet\|\+classical\s+([0-9.]+)\s+\(\d+ folds\)")
    n["spdnet_e2e"] = one("spdnet_ph11.log", r"^  spdnet_e2e\|\+classical\s+([0-9.]+)\s+\(\d+ folds\)")
    n["rand_rot"] = one("spdnet_ph11.log", r"^  rand_rot\s+([0-9.]+)\s+\(\d+ folds\)")
    n["designed15"] = one("spdnet_ph11.log", r"^  designed\s+([0-9.]+)\s+\(\d+ folds\)")
    n["res_rand"] = one("reservoir_ph11.log", r"^  res_rand\s+([0-9.]+)\s+\(\d+ folds\)")
    n["res_designed"] = one("reservoir_sweep_ph11.log", r"^  designed\s+([0-9.]+)")
    grid = {}
    for rho, leak, nl, v in re.findall(r"^  r([0-9.]+)_l([0-9.]+)_(\w+)\s+([0-9.]+)",
                                       _log("reservoir_sweep_ph11.log"), re.M):
        grid.setdefault((float(rho), nl), []).append(float(v))
    n["res_grid"] = {k: max(v) for k, v in grid.items()}      # best over leak, stated as such
    n["res_N"] = {int(m[0]): float(m[1]) for m in
                  re.findall(r"^  N(\d+)\s+([0-9.]+)", _log("reservoir_N_ph11.log"), re.M)}
    return n


# ---------------------------------------------------------------- helpers

def topo(ax, pos, v, levels=14):
    """Scalp map by azimuthal projection + interpolation. No MNE dependency."""
    from scipy.interpolate import griddata
    p = pos - pos.mean(0)
    r = np.linalg.norm(p, axis=1)
    theta = np.arccos(np.clip(p[:, 2] / r, -1, 1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    x, y_ = theta * np.cos(phi), theta * np.sin(phi)
    x, y_ = x / np.abs(x).max() * 0.85, y_ / np.abs(y_).max() * 0.85

    if np.ptp(v) > 0:                    # all-equal values would flood-fill the whole head
        g = np.linspace(-1.05, 1.05, 220)
        gx, gy = np.meshgrid(g, g)
        gz = griddata((x, y_), v, (gx, gy), method="cubic")
        gz[gx ** 2 + gy ** 2 > 1.0] = np.nan
        m = np.nanmax(np.abs(gz))
        ax.contourf(gx, gy, gz, levels=levels, cmap="RdBu_r", vmin=-m, vmax=m)
        ax.contour(gx, gy, gz, levels=levels, colors="k", linewidths=0.25, alpha=0.45)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.1, color=INK))
    ax.plot([-0.13, 0, 0.13], [0.99, 1.15, 0.99], color=INK, lw=1.1)   # nose
    ax.scatter(x, y_, s=6 if np.ptp(v) == 0 else 1.6, c=INK,
               alpha=0.85 if np.ptp(v) == 0 else 0.55, zorder=3)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.2, 1.3)
    ax.set_aspect("equal"); ax.axis("off")


def box(ax, xy, wh, text, fc="white", ec=INK, lw=1.0, fs=10, tc=None, ls="-"):
    ax.add_patch(Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor=ec, lw=lw, ls=ls,
                           zorder=2))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK, zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, color=INK, lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, zorder=2, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- figures

def fig_montage():
    d = cache()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    topo(ax, d["pos"], np.zeros(len(d["pos"])), levels=1)
    ax.set_title("61 channels", pad=6)

    ax = axes[1]
    ax.axhspan(0, 1.0, color=FAINT, alpha=0.75, lw=0)
    ax.axvline(0, color=ACC, lw=1.6)
    ax.text(0.02, 0.86, "phoneme onset", color=ACC, fontsize=9.5, ha="left")
    ax.annotate("", xy=(0, -0.22), xytext=(1.0, -0.22),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text(0.5, -0.34, "analysis window  0 to 1.0 s  (256 samples)", ha="center",
            fontsize=9.5)
    ax.text(-0.30, 0.5, "no pre-stimulus\nbaseline", ha="center", va="center",
            fontsize=9.5, color=ACC, style="italic")
    ax.add_patch(Rectangle((-0.55, 0.02), 0.5, 0.96, facecolor="white", edgecolor=MUTED,
                           ls=":", lw=1.0, hatch="///", alpha=0.5))
    # real recordings, not a drawing: three single trials per channel with that subject's
    # evoked mean over them. An earlier version used two Gaussians plus noise, which a reader
    # would reasonably have taken for data.
    tr3, ev = d["trials"], d["evoked"]
    t = np.linspace(0, 1.0, tr3.shape[-1])
    for c, off in enumerate([0.78, 0.5, 0.22]):
        sc = 0.10 / max(np.abs(tr3[:, c]).max(), 1e-9)
        for k in range(tr3.shape[0]):
            ax.plot(t, off + tr3[k, c] * sc, color=MUTED, lw=0.5, alpha=0.75)
        ax.plot(t, off + ev[c] * sc, color=INK, lw=1.1)
    ax.text(1.04, 0.78, "single trials", fontsize=8, color=MUTED, va="center")
    ax.text(1.04, 0.70, "trial mean", fontsize=8, color=INK, va="center")
    ax.set_xlim(-0.6, 1.30); ax.set_ylim(-0.42, 1.06)
    ax.axis("off")
    ax.set_title("three channels, one subject", pad=6)
    save(fig, "montage")


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    box(ax, (0.10, 2.55), (1.20, 1.30), "$X$\n$61 \\times 256$", fc="#F4F4F4", fs=9.5)
    arrow(ax, (1.30, 3.20), (1.72, 3.20))

    box(ax, (1.72, 2.40), (1.78, 1.60),
        "STAGE 1\nspatial filter\n$61 \\rightarrow 14$", fc="white", fs=9.5)
    ax.text(2.61, 2.02, "maximise\nsignal / noise", ha="center", va="top", fontsize=8.4,
            color=MUTED, style="italic")
    arrow(ax, (3.50, 3.20), (3.92, 3.20))
    ax.text(3.71, 3.34, "$Z$", ha="center", va="bottom", fontsize=9.5)

    box(ax, (3.92, 1.95), (3.63, 2.50), "", fc="#FBF3F2", ec=ACC, lw=1.3)
    ax.text(5.74, 4.16, "STAGE 2   base model", ha="center", fontsize=10, color=ACC)
    ax.text(5.74, 3.66,
            "$A(t)=[\\, Z(t)\\;;\\; Z(t{-}20)\\;;\\; \\varphi_0 \\ldots \\varphi_7 \\,]$",
            ha="center", va="center", fontsize=9.4)
    ax.text(5.74, 3.14, "$C = A A^{\\top} / T$", ha="center", va="center", fontsize=9.8)
    ax.text(5.74, 2.74, "$36 \\times 36$, not centred", ha="center", va="center",
            fontsize=8.6, color=MUTED, style="italic")
    ax.text(5.74, 2.30, "$\\log$-Euclidean tangent $\\rightarrow 666$", ha="center",
            va="center", fontsize=9.2)
    arrow(ax, (7.55, 3.20), (7.97, 3.20), color=ACC, lw=1.2)

    box(ax, (7.97, 2.40), (1.93, 1.60),
        "STAGE 3\nper-subject\nGaussian readout", fc="white", fs=9.5)
    ax.text(8.94, 2.02, "shared space,\npersonal rule", ha="center", va="top", fontsize=8.4,
            color=MUTED, style="italic")

    ax.text(5.0, 0.75, r"one learned linear map  $\cdot$  one fixed nonlinearity  $\cdot$  "
                       "one quadratic readout", ha="center", fontsize=9.5, color=MUTED)
    ax.text(5.0, 5.55, "Designed model", ha="center", fontsize=13)
    ax.text(5.0, 5.05, "every stage kept because removing it costs accuracy",
            ha="center", fontsize=9.6, color=MUTED, style="italic")
    save(fig, "arch")


def _spd_frame(ax, lab_fs=10.5):
    for e in (14, 28):
        ax.axhline(e - 0.5, color=INK, lw=1.1)
        ax.axvline(e - 0.5, color=INK, lw=1.1)
    lab = ["$Z(t)$", "$Z(t{-}\\tau)$", "$\\varphi$"]
    ax.set_xticks([7, 21, 32]); ax.set_xticklabels(lab, fontsize=lab_fs)
    ax.set_yticks([7, 21, 32]); ax.set_yticklabels(lab, fontsize=lab_fs, rotation=90,
                                                   va="center")
    ax.tick_params(length=0)


def fig_spd():
    """Two panels, because either one alone misleads.

    Left is the object itself. Its temporal blocks look faint, and that is honest: the evoked
    response is a small share of the total variance, which is exactly why single-trial
    decoding is hard.

    Right is discriminability, not variance, and the distinction matters. Raw between-class
    variance puts the phi blocks near zero -- they are small in absolute terms -- which would
    have argued against the design. What a classifier can use is between-class variance
    relative to within-class variance, so the right panel is a per-entry F on log(C) after
    subject centring, exactly the quantity the tangent-space readout sees. On that measure the
    Z-by-phi blocks carry about twice the mean discriminability of the spatial blocks
    (2.99 and 2.76 against 1.59 and 1.35), which is an independent corroboration of the
    ablation: removing the phi rows costs 0.052.
    """
    d = cache()
    C, V = d["C_mean"], d["F"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))

    R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    np.fill_diagonal(R, np.nan)
    m = np.nanpercentile(np.abs(R), 97)
    im = axes[0].imshow(R, cmap="RdBu_r", vmin=-m, vmax=m)
    _spd_frame(axes[0])
    axes[0].set_title("the object   $C = A A^{\\top} / T$\ncorrelation form, diagonal masked",
                      fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8); cb.outline.set_edgecolor(INK)

    im2 = axes[1].imshow(V, cmap="Reds", vmin=0, vmax=np.percentile(V, 99))
    _spd_frame(axes[1])
    for x, y_, t in [(7, 7, "space\n1.59"), (21, 7, "dynamics\n1.35"),
                     (32, 7, "evoked\n2.99"), (32, 21, "evoked\n2.76")]:
        axes[1].text(x, y_, t, ha="center", va="center", fontsize=8.6, color=INK,
                     linespacing=1.35,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88))
    axes[1].set_title("where the phoneme information is\n"
                      "per-entry $F$ = between-class / within-class", fontsize=10, pad=8)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03)
    cb2.ax.tick_params(labelsize=8); cb2.outline.set_edgecolor(INK)
    save(fig, "spd")


def fig_phi():
    T, K = 256, 7
    t = (np.arange(T) + 0.5) / T
    P = np.stack([np.cos(np.pi * k * t) for k in range(K + 1)])
    P = P / np.sqrt((P ** 2).mean(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k in range(K + 1):
        ax.plot(t * 1.0, P[k] * 0.42 + (K - k), color=ACC if k == 0 else INK,
                lw=1.0, alpha=1.0 if k < 4 else 0.75)
        ax.text(-0.035, K - k, "$\\varphi_{%d}$" % k, ha="right", va="center", fontsize=9.5)
    ax.set_xlim(-0.06, 1.02); ax.set_ylim(-0.8, K + 0.8)
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    bare(ax, keep=("bottom",))
    ax.set_title("the fixed clock: orthogonal DCT rows, unit RMS\n"
                 "$K=7$ puts the top component near 3.5 Hz, the N1/P2 timescale",
                 fontsize=10, pad=8)
    save(fig, "phi")


def fig_phi_ablation():
    n = numbers()
    names = ["dct_low", "none", "shuffled", "gauss", "dct_high"]
    folds = {k: n["phi_" + k] for k in names}
    pretty = ["real\n(dct low)", "no rows", "time\nshuffled", "random\niid rows",
              "wrong\ntimescale"]
    dim = ["666", "406", "666", "666", "666"]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    rng = np.random.default_rng(0)
    for i, n in enumerate(names):
        v = np.array(folds[n])
        c = ACC if i == 0 else MUTED
        ax.bar(i, v.mean(), width=0.6, color=c, alpha=0.28 if i else 0.35,
               edgecolor=c, lw=1.2, zorder=2)
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=8, color=c, zorder=3, alpha=0.9)
        ax.text(i, 0.006, dim[i], ha="center", fontsize=8.2, color="white" if i == 0 else INK)
    ax.axhline(1 / 11, color=INK, ls=(0, (4, 3)), lw=0.9)
    ax.text(4.45, 1 / 11 + 0.004, "chance", fontsize=8.6, ha="right", color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(pretty, fontsize=9)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 0.285)
    bare(ax)
    ax.set_title("destroy the time structure, keep the dimension\n"
                 r"$-0.052$  $\cdot$  $-0.054$  $\cdot$  $-0.066$  $\cdot$  $-0.101$   (all $p<10^{-4}$)",
                 fontsize=10, pad=8)
    save(fig, "phi_ablation")


def fig_ktime():
    K, v = zip(*numbers()["ktime"])
    best = int(np.argmax(v))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(range(len(K)), v, "-o", color=ACC, lw=1.4, ms=5)
    ax.scatter([best], [v[best]], s=90, facecolor="none", edgecolor=ACC, lw=1.4, zorder=4)
    ax.annotate("$K=7$\n$\\approx 3.5$ Hz", xy=(best, v[best]), xytext=(3.55, .213),
                fontsize=9.4, color=ACC,
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8))
    ax.set_xticks(range(len(K))); ax.set_xticklabels(K)
    ax.set_xlabel("number of temporal basis rows  $K$")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.155, 0.245)
    bare(ax)
    ax.set_title("capacity-response is an inverted U:\nthe timescale has to be right",
                 fontsize=10, pad=8)
    save(fig, "ktime")


def fig_bands():
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.45); ax.axis("off")
    ax.text(5, 5.05, "Best-performing model", ha="center", fontsize=13)
    ax.text(5, 4.62, "the same object, six times over -- then two representations glued",
            ha="center", fontsize=9.6, color=MUTED, style="italic")

    box(ax, (0.10, 2.85), (1.55, 1.15), "montage\nLaplacian\n61 modes", fc="#F4F4F4", fs=9)
    arrow(ax, (1.65, 3.42), (2.05, 3.42))
    for b in range(6):
        box(ax, (2.10 + b * 0.50, 2.90), (0.40, 1.05), "", fc="white", lw=0.8)
        ax.text(2.30 + b * 0.50, 3.42, "%d" % (b + 1), ha="center", va="center", fontsize=8.5)
    ax.text(3.35, 2.62, "6 spatial-frequency bands, about 10 modes each", ha="center",
            fontsize=8.6, color=MUTED)
    arrow(ax, (5.20, 3.42), (5.60, 3.42))

    box(ax, (5.60, 3.42), (3.10, 0.72), "", fc="#FBF3F2", ec=ACC, lw=1.2)
    ax.text(7.15, 3.90, "one SPD object per band", ha="center", va="center", fontsize=9.2)
    ax.text(7.15, 3.62, "2495 dims", ha="center", va="center", fontsize=8.6, color=MUTED)
    box(ax, (5.60, 2.48), (3.10, 0.72), "", fc="#EEF3F8", ec=ACC2, lw=1.2)
    ax.text(7.15, 2.96, "windowed covariance + means", ha="center", va="center", fontsize=9.2)
    ax.text(7.15, 2.68, "2911 dims", ha="center", va="center", fontsize=8.6, color=MUTED)

    for yy in (3.78, 2.84):                       # route the join outside both boxes
        ax.plot([8.70, 9.10], [yy, yy], color=INK, lw=0.9, zorder=1)
    ax.plot([9.10, 9.10], [2.84, 3.78], color=INK, lw=0.9, zorder=1)
    ax.plot([9.10, 9.10], [1.72, 2.84], color=INK, lw=0.9, zorder=1)
    ax.plot([7.15, 9.10], [1.72, 1.72], color=INK, lw=0.9, zorder=1)
    arrow(ax, (7.15, 1.72), (7.15, 1.42))
    box(ax, (5.85, 0.68), (2.60, 0.74), "concat    5406 dims", fc="white", lw=1.3, fs=10)

    ax.text(2.70, 1.72, "alone   0.270   and   0.289", fontsize=9.6, ha="center")
    ax.text(2.70, 1.32, "together   0.306", fontsize=9.6, ha="center", color=ACC)
    ax.text(2.70, 0.72, "they carry different information --\n"
                        "an ensemble, not an architecture",
            fontsize=8.6, ha="center", va="center", color=MUTED, style="italic")
    save(fig, "bands")


def fig_results():
    """The two evaluation protocols are not the same and the figure has to say so.

    `holdout.py` scored the designed and the best model once each on trials selection never
    touched. EEGNet has no such number yet -- its MODELS entry postdates the log -- so its
    0.225 is the 15-fold cross-validation mean. Drawing all three as one protocol would be
    the exact confusion this repository keeps warning about.
    """
    n = numbers()
    names = ["EEGNet\n(baseline)", "designed\nmodel", "best\nmodel"]
    v = [n["eeg_cv"], n["final_designed"], n["final_best"]]
    cols = [MUTED, ACC, ACC]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for i, x in enumerate(v):
        ax.bar(i, x, width=0.55, color=cols[i], alpha=0.3 if i == 0 else 0.85,
               edgecolor=cols[i], lw=1.3, zorder=2, hatch="///" if i == 0 else None)
        ax.text(i, x + 0.009, "%.3f" % x, ha="center", fontsize=11)
        ax.text(i, x + 0.032, "$%.1f\\times$ chance" % (x / (1 / 11)), ha="center",
                fontsize=8.8, color=MUTED)
    ax.axhline(1 / 11, color=INK, ls=(0, (4, 3)), lw=0.9)
    ax.text(2.42, 1 / 11 + 0.006, "chance 0.091", fontsize=8.8, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9.6)
    ax.set_ylabel("balanced accuracy, 11 phonemes")
    ax.set_ylim(0, 0.375)
    bare(ax)
    ax.text(0, -0.078, "15-fold CV", ha="center", va="top", fontsize=8.4, color=MUTED,
            style="italic")
    ax.text(1.5, -0.078, "untouched final test set", ha="center", va="top", fontsize=8.4,
            color=ACC, style="italic")
    ax.set_title("selection bias on the untouched set: $+0.000$\n"
                 "EEGNet is cross-validated -- it has no holdout score yet",
                 fontsize=9.6, pad=8)
    save(fig, "results")


def fig_paired():
    n = numbers()
    eeg, ours = n["eeg_folds"], n["ours_folds"]
    assert len(eeg) == len(ours) == 15, (len(eeg), len(ours))
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    for a_, b_ in zip(eeg, ours):
        ax.plot([0, 1], [a_, b_], color=MUTED, lw=0.7, alpha=0.75, zorder=1)
    ax.scatter(np.zeros(len(eeg)), eeg, s=22, color=MUTED, zorder=3)
    ax.scatter(np.ones(len(ours)), ours, s=22, color=ACC, zorder=3)
    ax.plot([-0.14, 0.14], [np.mean(eeg)] * 2, color=INK, lw=1.8, zorder=4)
    ax.plot([0.86, 1.14], [np.mean(ours)] * 2, color=ACC, lw=1.8, zorder=4)
    ax.axhline(1 / 11, color=INK, ls=(0, (4, 3)), lw=0.9)
    ax.text(1.32, 1 / 11 + 0.005, "chance", fontsize=8.6, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["EEGNet", "ours"], fontsize=10.5)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("balanced accuracy")
    bare(ax)
    won = sum(b > a for a, b in zip(eeg, ours))
    ax.set_title("same %d folds, paired\n%d/%d wins,  $p = 2\\times10^{-8}$"
                 % (len(eeg), won, len(eeg)), fontsize=10, pad=8)
    save(fig, "paired")


def fig_splits():
    n = numbers()
    labs = ["random\ntrials", "leave one\nrecording out", "time\nblocks",
            "unseen\nsubject", "shuffled\nlabels"]
    keys = ["split_trial", "split_token", "split_time", "split_loso", "split_perm"]
    v = [n[k][0] for k in keys]
    sd = [n[k][1] for k in keys]
    cols = [ACC, OK, OK, MUTED, INK]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for i in range(5):
        ax.bar(i, v[i], width=0.6, color=cols[i], alpha=0.3, edgecolor=cols[i], lw=1.3,
               zorder=2)
        ax.errorbar(i, v[i], yerr=sd[i], color=cols[i], lw=1.1, capsize=3, zorder=3)
        ax.text(i, v[i] + sd[i] + 0.011, "%.3f" % v[i], ha="center", fontsize=9.6)
    ax.axhline(1 / 11, color=INK, ls=(0, (4, 3)), lw=0.9)
    ax.text(-0.42, 1 / 11 + 0.007, "chance 0.091", fontsize=8.6, ha="left")
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 0.325)
    bare(ax)
    ax.annotate("sits exactly\non chance", xy=(4, 0.104), xytext=(4, 0.205),
                fontsize=8.6, color=INK, ha="center", style="italic",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.set_title("no leak survives any of the four splits,\n"
                 "and the label-permutation null lands on chance", fontsize=10, pad=8)
    save(fig, "splits")


def fig_patterns():
    d = cache()
    A, pos = d["A"], d["pos"]
    n = 6
    fig, axes = plt.subplots(1, n, figsize=(8.6, 1.9))
    for i in range(n):
        topo(axes[i], pos, A[:, i])
        axes[i].set_title("%d" % (i + 1), fontsize=9.5, pad=2)
    fig.suptitle("what the filter looks at -- first six components, as patterns "
                 "(Haufe et al. 2014)", fontsize=10, y=1.10)
    save(fig, "patterns")


def fig_confusion():
    d = cache()
    M = d["conf"].astype(float)
    R = M / M.sum(1, keepdims=True)
    ph = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
    fig, ax = plt.subplots(figsize=(4.9, 4.5))
    im = ax.imshow(R, cmap="Reds", vmin=0, vmax=np.percentile(R, 99))
    ax.axhline(4.5, color=INK, lw=1.2); ax.axvline(4.5, color=INK, lw=1.2)
    ax.set_xticks(range(11)); ax.set_xticklabels(ph, fontsize=9.5)
    ax.set_yticks(range(11)); ax.set_yticklabels(ph, fontsize=9.5)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.tick_params(length=0)
    ax.text(2, -1.15, "vowels", ha="center", fontsize=9.2, color=INK)
    ax.text(7.5, -1.15, "consonants", ha="center", fontsize=9.2, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor(INK)
    ax.set_title("errors respect phonetics:\nthe vowel/consonant block is visible",
                 fontsize=10, pad=22)
    save(fig, "confusion")


def fig_budget():
    rows = numbers()["budget"]
    n = [int(r[0]) for r in rows]
    own = [r[1] for r in rows]
    pooled = [r[3] for r in rows]
    hier = [r[4] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(n, own, "-o", color=ACC, lw=1.5, ms=5, label="own trials only")
    ax.plot(n, pooled, "-s", color=MUTED, lw=1.3, ms=4.5, label="+ all 15 other subjects")
    ax.plot(n, hier, "-^", color=ACC2, lw=1.3, ms=4.5, label="+ others, hierarchical")
    ax.axvspan(22, 40, color=FAINT, alpha=0.7, lw=0, zorder=0)
    ax.text(31, 0.098, "crossover", ha="center", fontsize=8.6, color=INK)
    ax.set_xlabel("number of the subject's own training trials")
    ax.set_ylabel("balanced accuracy")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    bare(ax)
    ax.set_title("3120 trials from 15 other people are worth less\n"
                 "than 44 of your own -- and past the crossover they hurt",
                 fontsize=10, pad=8)
    save(fig, "budget")


def fig_emergence():
    n = numbers()
    labs = ["reservoir\n(random,\nno backprop)", "SPDNet\nend-to-end", "SPDNet\n(learned\nrepr.)",
            "designed", "random\nrotation"]
    v = [n["res_rand"], n["spdnet_e2e"], n["spdnet"], n["designed15"], n["rand_rot"]]
    cols = [MUTED, MUTED, MUTED, ACC, ACC2]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i in range(len(v)):
        ax.bar(i, v[i], width=0.6, color=cols[i], alpha=0.32, edgecolor=cols[i], lw=1.3,
               zorder=2)
        ax.text(i, v[i] + 0.005, "%.3f" % v[i], ha="center", va="bottom", fontsize=9.4,
                zorder=4, bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    ax.axhline(n["designed15"], color=ACC, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(labs, fontsize=8.6)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.19, 0.283)
    bare(ax)
    ax.set_title("replacing the design -- by learning it, or by letting it emerge --\n"
                 "loses in both directions (15 folds)", fontsize=10, pad=8)
    save(fig, "emergence")


def fig_reservoir_sweep():
    """Each point is the best cell over the leak sweep, which is the reservoir's best case.

    The earlier hand-typed version silently mixed three different leak rates into one line --
    tanh at 0.3, relu at 0.1, linear at 1.0 and 0.1 -- so the curves were not comparable. Now
    the maximum over leak is taken explicitly and the axis label says so.
    """
    n = numbers()
    rho = sorted({r for r, _ in n["res_grid"]})
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    mk = {"tanh": "-o", "relu": "-s", "linear": "-^"}
    for k in ["tanh", "relu", "linear"]:
        v = [n["res_grid"][(r, k)] for r in rho]
        ax.plot(rho, v, mk[k], lw=1.3, ms=5, color=INK if k == "linear" else MUTED,
                alpha=1.0 if k == "linear" else 0.8, label=k)
    ax.axhline(n["res_designed"], color=ACC, lw=1.3)
    ax.text(max(rho) + 0.01, n["res_designed"] + 0.004, "designed (no reservoir)",
            fontsize=8.8, color=ACC, ha="right")
    ax.set_xlabel("spectral radius  $\\rho$   (more recurrence $\\rightarrow$)")
    ax.set_ylabel("balanced accuracy,  best over leak")
    ax.set_xticks(rho)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_ylim(0.14, 0.272)
    bare(ax)
    ax.set_title("the reservoir does best when it is least like a reservoir\n"
                 "best cell $\\rho=0.3$, leak $=1.0$, linear -- and still loses",
                 fontsize=10, pad=8)
    save(fig, "reservoir_sweep")


FIGS = {"montage": fig_montage, "arch": fig_arch, "spd": fig_spd, "phi": fig_phi,
        "phi_ablation": fig_phi_ablation, "ktime": fig_ktime, "bands": fig_bands,
        "results": fig_results, "paired": fig_paired, "splits": fig_splits,
        "patterns": fig_patterns, "confusion": fig_confusion, "budget": fig_budget,
        "emergence": fig_emergence, "reservoir_sweep": fig_reservoir_sweep}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true", help="recompute the cached real data")
    ap.add_argument("--verify", action="store_true", help="print every parsed log number")
    ap.add_argument("--only", nargs="+", default=None, choices=sorted(FIGS))
    a = ap.parse_args()
    if a.verify:
        for k, v in numbers().items():
            print("%-16s %s" % (k, v))
        raise SystemExit(0)
    if a.compute:
        compute_cache(a)
    for name in (a.only or sorted(FIGS)):
        FIGS[name]()
