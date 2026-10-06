
"""
Which phonetic distinctions live in a shared brain geometry, and which are personal?

The repo's pooling failures are always reported for the whole 11-way problem, which mixes
distinctions that may behave very differently. This measures each phonetic contrast on its
own, in the space where classification actually happens (the whitened feature space), with
two quantities per contrast:

  1. sharing of the decision rule
       w_s = Sigma^-1 (m_s1 - m_s0)     the subject's discriminant direction
       between = mean pairwise cos(w_s, w_t) over subject pairs
       within  = mean cos(w_s^halfA, w_s^halfB) from split-half of the same subject
       shared  = between / within        (noise-corrected: 1.0 = fully shared geometry)

     This is the classifier-space version of the RDM accounting in shared_ceiling.py. It is
     the right version, because stagewise.py showed the failure is in the decision rule, not
     in the feature space, and an RDM correlation does not see decision rules.

  2. how much pooling actually pays
       accuracy of the hierarchical model at g = 0 (personal means), g = 1 (group means)
       and at the inner-CV optimum, on identical folds.

Prediction being tested: coarse contrasts (consonant/vowel, place) are shared and should
prefer large g; fine contrasts (voicing pairs, vowel identity) are personal and should
prefer g near 0. If that holds, the 11-way model should pool per contrast rather than
globally -- which is a different architecture from anything tried so far.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, PHON, blend, cell_means, gauss_predict, lda_pooled, load_pooled,
                    pca_blocks, shared_precision)

# name -> (class 0 phonemes, class 1 phonemes, why this contrast exists)
CONTRASTS = {
    "C_vs_V":        (["a", "i", "e", "u", "o"], ["s", "z", "p", "b", "t", "d"]),
    "manner_stopfric": (["b", "p", "d", "t"], ["s", "z"]),
    "place_stops":   (["b", "p"], ["d", "t"]),
    "place_obstr":   (["b", "p"], ["d", "t", "s", "z"]),
    "voicing_all":   (["b", "d", "z"], ["p", "t", "s"]),
    "voicing_bp":    (["b"], ["p"]),
    "voicing_dt":    (["d"], ["t"]),
    "voicing_sz":    (["s"], ["z"]),
    "vowel_front":   (["i", "e"], ["u", "o"]),
    "vowel_high":    (["i", "u"], ["e", "o"]),
    "vowel_a_vs_hi": (["a"], ["i", "u"]),
}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def w_direction(Z, y, sid, idx, s, P):
    i = idx[sid[idx] == s]
    i1, i0 = i[y[i] == 1], i[y[i] == 0]
    if len(i1) < 3 or len(i0) < 3:
        return None
    return P @ (Z[i1].mean(0) - Z[i0].mean(0))


def sharing(Z, y, sid, idx, P, rng):
    subs = [s for s in np.unique(sid[idx])]
    W = {s: w_direction(Z, y, sid, idx, s, P) for s in subs}
    W = {s: w for s, w in W.items() if w is not None}
    ks = sorted(W)
    between = [cosine(W[a], W[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
    within = []
    for s in ks:
        i = idx[sid[idx] == s]
        ha, hb = [], []
        for c in (0, 1):
            ic = i[y[i] == c].copy()
            rng.shuffle(ic)
            h = len(ic) // 2
            ha.append(ic[:h]); hb.append(ic[h:2 * h])
        wa = w_direction(Z, y, sid, np.concatenate(ha), s, P)
        wb = w_direction(Z, y, sid, np.concatenate(hb), s, P)
        if wa is not None and wb is not None:
            within.append(cosine(wa, wb))
    b = float(np.mean(between)) if between else np.nan
    w = float(np.mean(within)) if within else np.nan
    return b, w, (b / w if w and w > 0.02 else np.nan)


def _fold(args):
    X, y11, sid, tr, te, d, K, p, tau, n_sub, grid, seed = args
    Z = pca_blocks(X, tr, d, K, p, tau, sid=sid, subject_center=True)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (g0, g1) in CONTRASTS.items():
        idx0 = np.isin(y11, [PHON.index(v) for v in g0])
        idx1 = np.isin(y11, [PHON.index(v) for v in g1])
        keep = idx0 | idx1
        yb = np.where(idx1, 1, 0)
        trc = tr[keep[tr]]; tec = te[keep[te]]
        if len(tec) < 2 * n_sub:
            continue
        M, N, Mp = cell_means(Z, yb, sid, trc, 2, n_sub)
        P = shared_precision(Z, yb, sid, trc, M)
        # controls for what shrinking toward the group actually does:
        #   perm  group means with the class labels swapped -> keeps the group's norm and
        #         geometry but destroys its class information
        #   zero  shrink toward the origin -> pure variance reduction, no group at all
        Mperm = Mp[::-1].copy()
        Mzero = np.zeros_like(Mp)
        acc, accp, accz = {}, {}, {}
        for g in grid:
            acc[g] = balanced_accuracy_score(
                yb[tec], gauss_predict(Z, tec, sid, blend(M, Mp, g), P))
            accp[g] = balanced_accuracy_score(
                yb[tec], gauss_predict(Z, tec, sid, blend(M, Mperm, g), P))
            accz[g] = balanced_accuracy_score(
                yb[tec], gauss_predict(Z, tec, sid, blend(M, Mzero, g), P))
        b, w, frac = sharing(Z, yb, sid, trc, P, rng)
        out[name] = dict(acc=acc, acc_perm=accp, acc_zero=accz,
                         between=b, within=w, shared=frac,
                         n=int(keep.sum()),
                         pooled_sk=lda_pooled(Z, yb, trc, tec))
    return out


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled('ph11', a.data)
    print("pooled trials=%d subjects=%d d=%d K=%d" % (len(X), n_sub, a.d, a.windows),
          flush=True)

    grid = np.linspace(0, 1, 11)
    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, y, sid, tr, te, a.d, a.windows, a.p, a.tau, n_sub, grid, sd))

    acc = {k: {g: [] for g in grid} for k in CONTRASTS}
    accp = {k: {g: [] for g in grid} for k in CONTRASTS}
    accz = {k: {g: [] for g in grid} for k in CONTRASTS}
    shr = {k: [] for k in CONTRASTS}
    btw = {k: [] for k in CONTRASTS}
    wth = {k: [] for k in CONTRASTS}
    nn = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k, v in out.items():
                for g in grid:
                    acc[k][g].append(v["acc"][g])
                    accp[k][g].append(v["acc_perm"][g])
                    accz[k][g].append(v["acc_zero"][g])
                shr[k].append(v["shared"]); btw[k].append(v["between"])
                wth[k].append(v["within"]); nn[k] = v["n"]
            print("  fold done", flush=True)

    print()
    print("=== per-contrast sharing and pooling gain (balanced acc, chance 0.500) ===")
    print("%-16s %6s  %7s %7s %7s  %6s %7s %7s  %6s %6s %6s" % (
        "contrast", "n", "g=0", "g=1", "best", "g*", "PERM@g*", "ZERO@g*",
        "cos_b", "cos_w", "shared"))
    rows = []
    for k in CONTRASTS:
        if not shr.get(k):
            continue
        a0 = np.mean(acc[k][grid[0]]); a1 = np.mean(acc[k][grid[-1]])
        means = np.array([np.mean(acc[k][g]) for g in grid])
        gstar = grid[int(means.argmax())]
        rows.append((k, nn[k], a0, a1, means.max(), gstar,
                     np.mean(accp[k][gstar]), np.mean(accz[k][gstar]),
                     np.nanmean(btw[k]), np.nanmean(wth[k]), np.nanmean(shr[k])))
    for r in sorted(rows, key=lambda r: -r[10] if np.isfinite(r[10]) else 1e9):
        print("%-16s %6d  %7.3f %7.3f %7.3f  %6.1f %7.3f %7.3f  %6.3f %6.3f %6.3f" % r)
    print()
    print("  g=0 personal class means | g=1 group class means | both on a shared covariance")
    print("  cos_b  mean pairwise cosine between subjects' discriminant directions")
    print("  cos_w  within-subject split-half cosine, the noise ceiling for cos_b")
    print("  shared cos_b / cos_w")
    print("  PERM   same shrinkage toward group means whose class labels were swapped")
    print("  ZERO   same shrinkage toward the origin, i.e. no group information at all")
    print("  if PERM and ZERO match the real number, gamma is regularisation, not sharing")
    print()
    for k in CONTRASTS:
        if not shr.get(k):
            continue
        print("  %-16s acc vs g: %s" % (k, "  ".join(
            "%.1f:%.3f" % (g, np.mean(acc[k][g])) for g in grid)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
