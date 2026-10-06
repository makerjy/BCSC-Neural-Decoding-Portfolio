
"""
The best-performing model, which is a concatenation and therefore not a design.

Two representations are built per spatial band and glued together:

    band_one     one SPD object per band -- [Z(t); Z(t-tau); phi_0..phi_7], not centred, so
                 the temporal rows keep the evoked waveform. Space, cross-lag dynamics and
                 time order interact inside one matrix.
    band_blocks  the separate-then-concatenate pipeline -- delay-augmented tangent plus
                 per-window tangent and per-window mean. Time order lives in the window
                 boundaries rather than in the object.

Alone, band_one loses to band_blocks within subject (0.270 vs 0.289, p<1e-4) and ties it
pooled; concatenated they beat band_blocks by +0.017 per-subject (p=0.0017) and +0.026 pooled
(p<1e-4). So the two carry different information and gluing them wins -- which is an ensemble
argument, not an architectural one. `final_model.py` is the designed model; this file exists
because it is the highest number the session produced and hiding that would be dishonest.

    untouched final test set   0.306   (SE 0.018, selection bias +0.000)
    cross-validation, 15 folds 0.305
    EEGNet pooled+ft_head      0.225   (same folds, 15/15 losses, p=2e-8)
    unseen subject (LOSO)      0.143   -- does not work, same as every other model

Cost of the extra performance: 5406 dimensions against final_model's 666, and no component
here was selected by a mechanism control. Use final_model.py when the question is "which
parts are justified", use this when the question is "what is the ceiling we reached".
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import ttest_1samp

from sdcore import (CHANCE, DATA, basis_full, block_features, folds_for, laplacian_modes,
                    lda_persubj, lda_pooled, load_pooled, tangent, unified_spd,
                    window_blocks)


def features(X, y, sid, fit, V, wlap, a):
    """band_one and band_blocks per band, concatenated. Everything fitted uses `fit` only."""
    U = basis_full("lap", X, fit, V, wlap, 0.0, 0.5)
    edges = np.linspace(0, X.shape[1], a.bands + 1).astype(int)
    one, blk = [], []
    for b in range(a.bands):
        Zb = np.einsum("ck,nct->nkt", U[:, edges[b]:edges[b + 1]], X)
        one.append(tangent(unified_spd(Zb, fit, a.p, a.tau, a.K, a.phi), fit, sid))
        blk.append(block_features(window_blocks(Zb, fit, a.windows, a.p, a.tau), fit, sid))
    return np.hstack(one + blk)


def _job(args):
    (X, y, sid, tr, te, n_cls, a, V, wlap, loso, perm_seed, readout) = args
    if perm_seed is not None:                      # shuffle labels within subject
        rng = np.random.default_rng(perm_seed)
        y = y.copy()
        for s in np.unique(sid):
            i = np.where(sid == s)[0]
            y[i] = rng.permutation(y[i])
    F = features(X, y, sid, tr, V, wlap, a)
    if readout == "dim":
        return float(F.shape[1])
    if loso or readout == "pooled":
        return lda_pooled(F, y, tr, te)            # unseen subject: no personal readout
    return lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    V, wlap = laplacian_modes(pos)                 # fixed montage: computed once
    n_cls = len(np.unique(y)); chance = CHANCE[a.task]
    print("task=%s trials=%d subjects=%d classes=%d | study=%s bands=%d p=%d tau=%d K=%d "
          "windows=%d" % (a.task, len(X), n_sub, n_cls, a.study, a.bands, a.p, a.tau, a.K,
                          a.windows), flush=True)

    def folds(kind):
        return folds_for(kind, X, y, tok, sid, a.seeds)

    def base(kind, perm_seed=None, readout="persubj"):
        return [(X, y, sid, tr, te, n_cls, a, V, wlap, kind == "loso", perm_seed, readout)
                for tr, te in folds(kind)]

    jobs, tags = [], []
    if a.study == "perm":
        jobs += base("trial"); tags += ["real"] * len(folds("trial"))
        for r in range(a.perms):
            jobs += base("trial", perm_seed=1000 + r)
            tags += ["perm%d" % r] * len(folds("trial"))
    elif a.study == "readout":
        for r in ["persubj", "pooled"]:
            jobs += base("trial", readout=r); tags += [r] * len(folds("trial"))
    else:
        for k in ["trial", "token", "time", "loso"]:
            jobs += base(k); tags += [k] * len(folds(k))

    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for t, v in zip(tags, ex.map(_job, jobs)):
            res.setdefault(t, []).append(v)
            print("  %-8s fold %d  %.3f" % (t, len(res[t]), v), flush=True)

    print()
    print("=== %s (chance %.3f)  study=%s ===" % (a.task, chance, a.study))
    if a.study == "perm":
        real = np.array(res["real"], float)
        null = np.concatenate([res[k] for k in res if k != "real"])
        _, p0 = ttest_1samp(real, chance)
        print("  real          %.3f +- %.3f   vs-chance p=%.7f" % (
            real.mean(), real.std(), p0))
        print("  permutation   %.3f +- %.3f   (%d fits, %d shuffles)" % (
            null.mean(), null.std(), len(null), a.perms))
        print("  real above every permutation fold: %s" % (real.min() > null.max()))
        print("  empirical p = %.4f" % ((np.sum(null >= real.mean()) + 1) / (len(null) + 1)))
    else:
        for k in res:
            v = np.array(res[k], float)
            _, p0 = ttest_1samp(v, chance)
            print("  %-8s %.3f +- %.3f  (%d folds)  vs-chance p=%.7f" % (
                k, v.mean(), v.std(), len(v), p0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--study", default="split", choices=["split", "perm", "readout"])
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--perms", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
