
"""
Hierarchical pooling: share the noise model, personalise the class means.

Measured motivation (this repo, logs/sw.log):

    pooled basis + per-subject classifier   0.230
    pooled basis + pooled classifier        0.168   (-0.079, p=0.002)
    per-subject basis + per-subject clf     0.246   (-0.016 vs pooled basis, ns)

so the *feature space* transfers between people and the *decision rule* does not. Eight
subject-normalisation front-ends have already been tried and all land at or below naive
pooling (subject centring, z-scoring, one-hot subject, adversarial invariance, Riemannian
recentring, CORAL, joint diagonalisation, Procrustes alignment). They all attack the
features, which the measurement above says is the wrong target.

This splits the classifier instead of the features:

    w_s = w_0 + v_s          (Evgeniou & Pontil 2004, regularised multi-task learning)

in its Gaussian/LDA form -- one within-class covariance estimated from every subject's
trials (the part that genuinely is shared: a 2662-trial estimate instead of 166), and class
means shrunk from the subject's own estimate toward the group estimate,

    m_sc(g) = g * m_c + (1 - g) * m_sc ,   g picked by inner CV on training trials only.

g = 1 is exactly pooled training and g = 0 is per-subject means on a shared noise model, so
the model contains both baselines. Tuned, it cannot lose to either -- no earlier pooled
attempt in this repo had that property.

Representation is the one the repo's ablations selected, each block for a stated reason:
  spatial   PCA d                      evoked response is low-rank in sensor space
  dynamics  delay-augmented covariance cross-lag structure (unified2: +0.024 within subject)
  time      K disjoint window covariances   time order is the single replicated predictor
  evoked    K window means              first-order deflection the covariance discards
                                        (best_model3: cov+mean > cov, +0.024, p<1e-4)

Note on the data: npz2 epochs start at t=0, so the BASE=(-0.2,-0.11) window used by the
older scripts is empty and no baseline subtraction happens. That is why the window-mean
block carries so much -- it is the only place the evoked DC offset survives.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, blend, cell_means, gauss_predict, lda_persubj, lda_pooled,
                    load_pooled, pca_blocks, personal_precision, persubj_pipeline, pick_gamma,
                    shared_precision, time_folds, token_folds)

WMIN, WMAX = 0.0, 1.01


# ---------------------------------------------------------------- data

# ---------------------------------------------------------------- representation

# ---------------------------------------------------------------- hierarchical LDA

# ---------------------------------------------------------------- baselines

# ---------------------------------------------------------------- one fold

def _fold(args):
    (X, y, tok, sid, tr, te, d, K, p, tau, n_cls, n_sub, grid, want_champ) = args
    min_te, min_tr = (n_cls, 3 * n_cls)
    out = {}

    Z = pca_blocks(X, tr, d, K, p, tau, sid=sid, subject_center=True)
    te = np.sort(te)

    out["pooled_LDA"] = lda_pooled(Z, y, tr, te)
    out["persubj_LDA"] = lda_persubj(Z, y, sid, tr, te, min_te, min_tr)

    M, N, Mp = cell_means(Z, y, sid, tr, n_cls, n_sub)
    P = shared_precision(Z, y, sid, tr, M)
    out["hier_g1_pooledmeans"] = balanced_accuracy_score(
        y[te], gauss_predict(Z, te, sid, blend(M, Mp, 1.0), P))
    out["hier_g0_personalmeans"] = balanced_accuracy_score(
        y[te], gauss_predict(Z, te, sid, blend(M, Mp, 0.0), P))

    g, curve = pick_gamma(Z, y, sid, tr, n_cls, n_sub, grid)
    out["hier_gCV"] = balanced_accuracy_score(
        y[te], gauss_predict(Z, te, sid, blend(M, Mp, g), P))
    out["gamma"] = g
    out["gamma_curve"] = curve

    Pp = personal_precision(Z, y, sid, tr, M, n_sub, Z.shape[1])
    out["hier_gCV_personalcov"] = balanced_accuracy_score(
        y[te], gauss_predict(Z, te, sid, blend(M, Mp, g), Pp, per_subject_prec=True))

    out["persubj_pipeline"] = (persubj_pipeline(X, y, sid, tr, te, d, K, p, tau,
                                                min_te, min_tr) if want_champ else np.nan)
    return out


# ---------------------------------------------------------------- driver

def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data)
    n_cls = len(np.unique(y))
    chance = 1.0 / n_cls
    print("task=%s split=%s trials=%d subjects=%d classes=%d d=%d K=%d p=%d tau=%d" % (
        a.task, a.split, len(X), n_sub, n_cls, a.d, a.windows, a.p, a.tau), flush=True)

    grid = np.linspace(0, 1, 11)
    jobs = []
    if a.split == "trial":
        strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
        for sd in range(a.seeds):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
                jobs.append((X, y, tok, sid, tr, te, a.d, a.windows, a.p, a.tau,
                             n_cls, n_sub, grid, bool(a.champion)))
    elif a.split == "token":
        for tr, te in token_folds(tok, y, sid):
            jobs.append((X, y, tok, sid, tr, te, a.d, a.windows, a.p, a.tau,
                         n_cls, n_sub, grid, bool(a.champion)))
    else:
        for tr, te in time_folds(sid):
            jobs.append((X, y, tok, sid, tr, te, a.d, a.windows, a.p, a.tau,
                         n_cls, n_sub, grid, bool(a.champion)))

    NAMES = ["persubj_pipeline", "persubj_LDA", "pooled_LDA",
             "hier_g1_pooledmeans", "hier_g0_personalmeans",
             "hier_gCV", "hier_gCV_personalcov"]
    R = {k: [] for k in NAMES}
    G, CURVE = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k in NAMES:
                R[k].append(out[k])
            G.append(out["gamma"]); CURVE.append(out["gamma_curve"])
            print("  fold  g=%.1f  " % out["gamma"] + "  ".join(
                "%s %.3f" % (k[:16], out[k]) for k in NAMES), flush=True)

    print()
    print("  gamma picked by inner CV: mean %.2f  (per fold %s)" % (
        np.mean(G), " ".join("%.1f" % v for v in G)))
    cv = np.mean(CURVE, axis=0)
    print("  inner-CV accuracy vs gamma: " + "  ".join(
        "%.1f:%.3f" % (g, v) for g, v in zip(grid, cv)))
    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    ref = np.array(R["persubj_pipeline"] if a.champion else R["persubj_LDA"])
    for k in sorted(NAMES, key=lambda k: -np.nanmean(R[k])):
        v = np.array(R[k], float)
        if np.all(np.isnan(v)):
            continue
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-22s %.3f  vs-chance p=%.7f  vs-ref %+.3f (p=%.4f) %s" % (
            k, v.mean(), p0, v.mean() - ref.mean(), pb,
            "**BEATS**" if pb < 0.05 and v.mean() > ref.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--split", default="trial", choices=["trial", "token", "time"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--champion", type=int, default=1)
    main(ap.parse_args())
