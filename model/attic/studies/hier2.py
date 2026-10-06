
"""
Two numbers for "what is shared between brains": one for the noise, one for the classes.

hier.py forced a single shared within-class covariance and lost to per-subject LDA even
with the class means shrunk toward the group (0.221 vs 0.235). So the noise geometry is
personal too -- pooling 2662 trials to estimate one whitener is worse than 166 trials to
estimate your own, because 16 subjects' noise covariances are 16 different objects and their
average fits none of them.

The fix is to shrink both, each with its own coefficient, and let inner CV say how much:

    Sigma_s(l) = (1 - l) Sigma_s + l Sigma_group          l = 1 shared noise, 0 personal
    m_sc(g)    = (1 - g) m_sc    + g m_c                  g = 1 shared classes, 0 personal
    then a ridge  (1 - k) Sigma + k mean(eig) I           k = conditioning, also tuned

The model contains pooled LDA (g=1, l=1), per-subject RDA (g=0, l=0) and every mixture, so
tuning cannot lose to either. (l*, g*) is the readout: it says, in the units the classifier
actually uses, which half of the subject difference matters.

Implementation note -- the (l, k) grid is free. Whiten by W = Sigma_group^-1/2 once; then
Sigma_s(l) = W^-1 [(1-l) W Sigma_s W + l I] W^-1, and every member of the family shares the
eigenvectors of W Sigma_s W. One eigendecomposition per subject covers the whole grid, and
scoring reduces to a diagonal reweighting in that basis.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, GAM, KAP, LAM, lda_persubj, lda_pooled, load_pooled, pca_blocks,
                    persubj_pipeline, prepare, score_grid)

def _fold(args):
    (X, y, sid, tr, te, d, K, p, tau, n_cls, n_sub, want_champ, inner, seed) = args
    min_te, min_tr = n_cls, 3 * n_cls
    Z = pca_blocks(X, tr, d, K, p, tau, sid=sid, subject_center=True)
    out = {}
    out["pooled_LDA"] = lda_pooled(Z, y, tr, te)
    out["persubj_LDA"] = lda_persubj(Z, y, sid, tr, te, min_te, min_tr)
    out["persubj_pipeline"] = (persubj_pipeline(X, y, sid, tr, te, d, K, p, tau,
                                                min_te, min_tr) if want_champ else np.nan)

    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid[tr], y[tr])])
    inner_surf = np.zeros((len(LAM), len(GAM), len(KAP))); nf = 0
    for itr, ite in StratifiedKFold(inner, shuffle=True, random_state=seed).split(tr, strat):
        M, Mp, basis = prepare(Z, y, sid, tr[itr], n_cls, n_sub)
        inner_surf += score_grid(Z, y, sid, tr[ite], M, Mp, basis, n_cls)
        nf += 1
    inner_surf /= nf
    li, gi, ki = np.unravel_index(inner_surf.argmax(), inner_surf.shape)

    M, Mp, basis = prepare(Z, y, sid, tr, n_cls, n_sub)
    full = score_grid(Z, y, sid, te, M, Mp, basis, n_cls)
    out["hier2_CV"] = float(full[li, gi, ki])
    out["hier2_oracle"] = float(full.max())
    out["hier2_pooled_corner"] = float(full[-1, -1, :].max())     # l=1, g=1
    out["hier2_personal_corner"] = float(full[0, 0, :].max())     # l=0, g=0
    out["lam"] = float(LAM[li]); out["gam"] = float(GAM[gi]); out["kap"] = float(KAP[ki])
    out["inner_surf"] = inner_surf
    out["outer_surf"] = full
    return out


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    print("task=%s trials=%d subjects=%d classes=%d d=%d K=%d p=%d tau=%d" % (
        a.task, len(X), n_sub, n_cls, a.d, a.windows, a.p, a.tau), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, y, sid, tr, te, a.d, a.windows, a.p, a.tau,
                         n_cls, n_sub, bool(a.champion), a.inner, sd))

    NAMES = ["persubj_pipeline", "persubj_LDA", "pooled_LDA", "hier2_CV",
             "hier2_oracle", "hier2_pooled_corner", "hier2_personal_corner"]
    R = {k: [] for k in NAMES}
    L, G, KP, SURF = [], [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k in NAMES:
                R[k].append(out[k])
            L.append(out["lam"]); G.append(out["gam"]); KP.append(out["kap"])
            SURF.append(out["outer_surf"])
            print("  fold  l=%.2f g=%.1f k=%.2f  " % (out["lam"], out["gam"], out["kap"]) +
                  "  ".join("%s %.3f" % (k[:15], out[k]) for k in NAMES), flush=True)

    print()
    print("  selected by inner CV:  lambda(noise sharing) %.2f   gamma(class sharing) %.2f"
          "   kappa %.2f" % (np.mean(L), np.mean(G), np.mean(KP)))
    S = np.mean(SURF, axis=0).max(-1)            # marginalise kappa
    print()
    print("  held-out accuracy surface (rows lambda = noise sharing, cols gamma = class sharing)")
    print("        " + "".join("%6.1f" % g for g in GAM))
    for li, l in enumerate(LAM):
        print("  l=%.2f " % l + "".join("%6.3f" % v for v in S[li]))
    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    ref = np.array(R["persubj_pipeline"] if a.champion else R["persubj_LDA"], float)
    for k in sorted(NAMES, key=lambda k: -np.nanmean(R[k])):
        v = np.array(R[k], float)
        if np.all(np.isnan(v)):
            continue
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-24s %.3f  vs-chance p=%.7f  vs-ref %+.3f (p=%.4f) %s" % (
            k, v.mean(), p0, v.mean() - ref.mean(), pb,
            "**BEATS**" if pb < 0.05 and v.mean() > ref.mean() else ""))
    print()
    print("  hier2_oracle is the best cell of the held-out surface, i.e. an upper bound that")
    print("  peeks at test data. hier2_CV is the honest number.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--champion", type=int, default=1)
    main(ap.parse_args())
