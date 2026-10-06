
"""
Electrode geometry as the subject/phoneme filter.

Three things the montage buys that the data-driven pipeline cannot:

(A) a subject-independent coordinate system.
    PCA is fitted, so "component 3" denotes a different scalp pattern for every subject and
    every fold, which makes "the group class mean" ill-defined -- the quantity hier.py had to
    shrink toward. The graph Laplacian eigenmodes of the montage are fixed by geometry alone:
    no estimation, no fold dependence, identical axes for all 16 subjects. spatial_bands.py
    already saw the consequence without naming it: pooled decoding on the Laplacian basis
    0.191 against 0.173 on PCA.

(B) the right null for shrinkage.
    Every regulariser in this repo pulls toward sigma^2 I, which asserts that electrodes are
    independent and exchangeable. Volume conduction says otherwise: neighbours share most of
    their signal. The physically correct null is a smooth field, whose precision is the
    Laplacian. Here that is a one-parameter spectral prewhitening, (lambda_i + eps)^(-alpha/2)
    per eigenmode: alpha > 0 favours smooth fields, alpha = 0 is the isotropic assumption the
    old pipeline made silently.

(D) sharing measured per spatial scale.
    Individual differences are mostly broad -- skull conductivity, cap placement, reference,
    alpha topography -- so they should live at low spatial frequency. Focal evoked sources
    live higher. spatial_bands.py tested this with a hard keep/drop rule on a cross-subject
    RDM and lost (0.169 vs 0.191). This is the soft version: each band keeps its own noise
    sharing lambda_b and class sharing gamma_b from hier2, tuned on training folds, and the
    band scores add.

Adding band scores is not a fusion heuristic. The eigenmodes are orthonormal, so under a
spatially stationary field the bands are uncorrelated and a block-diagonal covariance is the
correct model, which makes the summed log-likelihood exact rather than approximate. (Late
fusion of classifier outputs failed 4/4 in this repo; this is not that -- it sums scores
inside one likelihood, not decisions across models.)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


from sdcore import (DATA, band_scores, basis_full, block_features, laplacian_modes,
                    lda_persubj, lda_pooled, load_pooled, prepare, tune_head, window_blocks)

def _fold(args):
    (X, V, wlap, y, sid, tr, te, edges, K, p, tau, alpha, eps,
     n_cls, n_sub, inner, seed, do_global, basis_kind) = args
    U = basis_full(basis_kind, X, tr, V, wlap, alpha, eps)
    out = {"bands": []}

    Fb, tuned = [], []
    for bi in range(len(edges) - 1):
        Ub = U[:, edges[bi]:edges[bi + 1]]
        Zb = np.einsum("ck,nct->nkt", Ub, X)
        F = block_features(window_blocks(Zb, tr, K, p, tau), tr, sid)
        Fb.append(F)
        l, g, kk, inner_acc, _ = tune_head(F, y, sid, tr, n_cls, n_sub, inner, seed)
        M, Mp, basis = prepare(F, y, sid, tr, n_cls, n_sub)
        sc_te = band_scores(F, sid, te, M, Mp, basis, n_cls, l, g, kk)
        acc_te = balanced_accuracy_score(y[te], sc_te.argmax(1))
        tuned.append((sc_te, inner_acc))
        out["bands"].append(dict(
            band="%d-%d" % (edges[bi], edges[bi + 1]), lam=l, gam=g, kap=kk,
            inner=inner_acc, alone=acc_te,
            pooled=lda_pooled(F, y, tr, te),
            persubj=lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)))

    chance = 1.0 / n_cls
    Ssum = sum(s for s, _ in tuned)
    out["bandsum"] = balanced_accuracy_score(y[te], Ssum.argmax(1))
    wts = np.array([max(a - chance, 0.0) for _, a in tuned])
    if wts.sum() > 0:
        wts = wts / wts.sum() * len(wts)
        Sw = sum(w * s for w, (s, _) in zip(wts, tuned))
        out["bandsum_weighted"] = balanced_accuracy_score(y[te], Sw.argmax(1))
    else:
        out["bandsum_weighted"] = np.nan
    out["band_weights"] = wts

    Fall = np.hstack(Fb)
    out["pooled_LDA_allbands"] = lda_pooled(Fall, y, tr, te)
    out["persubj_LDA_allbands"] = lda_persubj(Fall, y, sid, tr, te, n_cls, 3 * n_cls)
    if do_global:                       # one (lambda, gamma) for all scales at once
        l, g, kk, _, _ = tune_head(Fall, y, sid, tr, n_cls, n_sub, inner, seed)
        M, Mp, basis = prepare(Fall, y, sid, tr, n_cls, n_sub)
        out["hier2_global"] = balanced_accuracy_score(
            y[te], band_scores(Fall, sid, te, M, Mp, basis, n_cls, l, g, kk).argmax(1))
        out["global_lgk"] = (l, g, kk)
    else:
        out["hier2_global"] = np.nan
        out["global_lgk"] = (np.nan,) * 3
    return out


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    V, wlap = laplacian_modes(pos)
    edges = np.linspace(0, X.shape[1], a.bands + 1).astype(int)
    print("task=%s trials=%d subjects=%d classes=%d | basis=%s | montage %d electrodes, "
          "laplacian eig %.2e..%.1f | bands %s | alpha=%.2f" % (
              a.task, len(X), n_sub, n_cls, a.basis, len(wlap), wlap[0], wlap[-1],
              list(edges), a.alpha), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, V, wlap, y, sid, tr, te, edges, a.windows, a.p, a.tau,
                         a.alpha, a.eps, n_cls, n_sub, a.inner, sd,
                         bool(a.global_model), a.basis))

    NAMES = ["bandsum", "bandsum_weighted", "hier2_global",
             "persubj_LDA_allbands", "pooled_LDA_allbands"]
    R = {k: [] for k in NAMES}
    BANDS = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k in NAMES:
                R[k].append(out[k])
            BANDS.append(out["bands"])
            print("  fold  " + "  ".join("%s %.3f" % (k[:14], out[k]) for k in NAMES),
                  flush=True)

    nb = len(BANDS[0])
    print()
    print("=== per band of the %s basis (lap: low index = smooth field, high = focal) ==="
          % a.basis)
    print("%-10s %8s %8s %8s | %6s %6s %6s" % (
        "band", "persubj", "pooled", "hier_alone", "lam*", "gam*", "kap*"))
    for b in range(nb):
        rows = [f[b] for f in BANDS]
        print("%-10s %8.3f %8.3f %8.3f | %6.2f %6.2f %6.2f" % (
            rows[0]["band"],
            np.mean([r["persubj"] for r in rows]),
            np.mean([r["pooled"] for r in rows]),
            np.mean([r["alone"] for r in rows]),
            np.mean([r["lam"] for r in rows]),
            np.mean([r["gam"] for r in rows]),
            np.mean([r["kap"] for r in rows])))
    print()
    print("  lam* = noise sharing (1 = one group covariance, 0 = each subject his own)")
    print("  gam* = class sharing (1 group class means, 0 personal class means)")
    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    ref = np.array(R["persubj_LDA_allbands"], float)
    for k in sorted(NAMES, key=lambda k: -np.nanmean(R[k])):
        v = np.array(R[k], float)
        if np.all(np.isnan(v)):
            continue
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-24s %.3f  vs-chance p=%.7f  vs-persubj %+.3f (p=%.4f) %s" % (
            k, np.nanmean(v), p0, np.nanmean(v) - ref.mean(), pb,
            "**BEATS**" if pb < 0.05 and np.nanmean(v) > ref.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--global_model", type=int, default=0)
    ap.add_argument("--basis", default="lap", choices=["lap", "pca", "chan", "rand"])
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
