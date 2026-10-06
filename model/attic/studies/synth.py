
"""
The synthesis: one melted object per spatial band.

Three measurements from today have to be reconciled.

  a) 61 spatial dimensions beat 16, and an ordering by scale beats an arbitrary one
     (C/V: pca61 0.752, lap61 0.753, chan61 0.724, rand61 0.724, pca16 0.704).
  b) The melted SPD object needs time order in it. Adding a temporal basis to the constant
     row took ph11 from 0.171 to 0.231, matching the whole separate-block pipeline (0.221)
     with fewer dimensions, and 0.255 when both are used.
  c) But the melted object is capped by timepoints: C = A A^T / T is estimated from T = 236
     samples, so its side must stay near 40, which with p = 2 delay copies and K + 1 = 8
     temporal rows leaves only 14 spatial modes. It cannot reach (a).

The cap is on the side of *one* matrix, not on how many matrices there are. So build one
melted object per spatial band: 10 modes per band gives side 2*10 + 8 = 28 against T = 236,
comfortably estimable, and six bands use all 61 dimensions. Inside a band, space, dynamics,
evoked deflection and time order still interact in a single matrix; across bands they are
concatenated and the classifier's Sigma^-1 couples them (geom.py measured that keeping that
coupling matters: block-diagonal bandsum 0.746 lost to joint 0.753).

Arms, identical folds:
    band_blocks     per band, window covariances + window means      == geom.py, today's best
    band_one        per band, one melted object with temporal basis  == the synthesis
    band_one+blocks both
    global_one      the single d=14 melted object                    == onemodel.py reference
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, basis_full, block_features, head, laplacian_modes, lda_persubj,
                    lda_pooled, load_pooled, loso_folds, tangent, unified_spd, window_blocks)

def _fold(args):
    (X, V, wlap, y, sid, tr, te, kind, edges, p, tau, K_time, phi, K_win,
     d_glob, lo_glob, n_cls, n_sub, inner, seed, want_head, want_comb) = args
    U = basis_full(kind, X, tr, V, wlap, 0.0, 0.5)
    reps = {}

    Fb_blocks, Fb_one = [], []
    for bi in range(len(edges) - 1):
        Zb = np.einsum("ck,nct->nkt", U[:, edges[bi]:edges[bi + 1]], X)
        Fb_blocks.append(block_features(window_blocks(Zb, tr, K_win, p, tau),
                                        tr, sid))
        Fb_one.append(tangent(unified_spd(Zb, tr, p, tau, K_time, phi), tr, sid))
    reps["band_blocks"] = np.hstack(Fb_blocks)
    reps["band_one"] = np.hstack(Fb_one)
    if want_comb:
        reps["band_one+blocks"] = np.hstack([reps["band_one"], reps["band_blocks"]])

    Zg = np.einsum("ck,nct->nkt", U[:, lo_glob:lo_glob + d_glob], X)
    reps["global_one"] = tangent(unified_spd(Zg, tr, p, tau, K_time, phi), tr, sid)

    out = {}
    for nm, F in reps.items():
        out[nm + "_persubj"] = lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
        out[nm + "_pooled"] = lda_pooled(F, y, tr, te)
        if want_head:
            acc, lgk = head(F, y, sid, tr, te, n_cls, n_sub, inner, seed)
            out[nm + "_hier2"] = acc
            out[nm + "_lgk"] = lgk
        else:
            out[nm + "_hier2"] = np.nan
            out[nm + "_lgk"] = (np.nan,) * 3
    out["dim"] = {k: v.shape[1] for k, v in reps.items()}
    return out


ARMS = ["band_blocks", "band_one", "band_one+blocks", "global_one"]
ARMS_NOCOMB = ["band_blocks", "band_one", "global_one"]


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    V, wlap = laplacian_modes(pos)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    edges = np.linspace(0, X.shape[1], a.bands + 1).astype(int)
    per_band = edges[1] - edges[0]
    print("task=%s trials=%d subjects=%d classes=%d | basis=%s bands=%d (%d modes each) "
          "| p=%d tau=%d K_time=%d -> melted side %d vs T~%d" % (
              a.task, len(X), n_sub, n_cls, a.basis, a.bands, per_band,
              a.p, a.tau, a.K_time, a.p * per_band + a.K_time + 1,
              X.shape[2] - (a.p - 1) * a.tau), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    if a.split == "loso":
        folds = loso_folds(sid)
    else:
        folds = [f for sd in range(a.seeds)
                 for f in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat)]
    jobs = []
    if True:
        for tr, te in folds:
            jobs.append((X, V, wlap, y, sid, tr, te, a.basis, edges, a.p, a.tau,
                         a.K_time, a.phi, a.windows, a.d_glob, a.lo_glob,
                         n_cls, n_sub, a.inner, 0, bool(a.head), bool(a.comb)))

    global ARMS
    ARMS = ARMS if a.comb else ARMS_NOCOMB
    KEYS = [x + s for x in ARMS for s in ("_persubj", "_pooled", "_hier2")]
    R = {k: [] for k in KEYS}
    LGK = {k: [] for k in ARMS}
    dims = None
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k in KEYS:
                R[k].append(out[k])
            for k in ARMS:
                LGK[k].append(out[k + "_lgk"])
            dims = out["dim"]
            print("  fold  " + "  ".join(
                "%s %.3f/%.3f" % (k[:11], out[k + "_persubj"], out[k + "_pooled"])
                for k in ARMS), flush=True)

    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    print("%-18s %6s %9s %9s %9s | %5s %5s" % (
        "representation", "dim", "persubj", "pooled", "hier2", "lam*", "gam*"))
    for k in ARMS:
        lg = np.mean(LGK[k], axis=0)
        print("%-18s %6d %9.3f %9.3f %9.3f | %5.2f %5.2f" % (
            k, dims[k], np.mean(R[k + "_persubj"]), np.mean(R[k + "_pooled"]),
            np.nanmean(R[k + "_hier2"]), lg[0], lg[1]))
    print()
    for suffix in ("_persubj", "_pooled", "_hier2"):
        ref = np.array(R["band_blocks" + suffix], float)
        if np.all(np.isnan(ref)):
            continue
        print("  --- %s (reference: band_blocks) ---" % suffix[1:])
        for k in sorted(ARMS, key=lambda k: -np.nanmean(R[k + suffix])):
            v = np.array(R[k + suffix], float)
            _, p0 = ttest_1samp(v, chance)
            _, pb = ttest_rel(v, ref)
            print("  %-18s %.3f  vs-chance p=%.7f  vs-ref %+.3f (p=%.4f) %s" % (
                k, v.mean(), p0, v.mean() - ref.mean(), pb,
                "**BEATS**" if pb < 0.05 and v.mean() > ref.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--basis", default="lap", choices=["lap", "pca", "chan", "rand"])
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K_time", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--d_glob", type=int, default=14)
    ap.add_argument("--lo_glob", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--comb", type=int, default=1)
    ap.add_argument("--split", default="trial", choices=["trial", "loso"])
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
