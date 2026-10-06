
"""
One object per trial: does putting time order inside it help, and does it replace the blocks?

unified.py built the melted version and it lost on its own (ph11 0.224 against 0.246 for
separate-then-concatenate, logs/u2.log); it only won when concatenated back on (0.273). The
reason is visible in its own construction:

    Z(t) = [x(t); x(t-tau); ...; x(t-(p-1)tau); 1]
    C    = (1/T) sum_t Z Z^T

the constant row keeps the window *mean*, and a covariance is an average over t, so time
order -- the one predictor this repo has confirmed nine times -- is not in the object at all.

The constant row is the k = 0 member of a temporal basis. Carry the rest (one_KT): the cross
blocks <mode_i, phi_k> are the projection of that mode's waveform onto temporal basis k, i.e.
the time-resolved evoked response, inside the same matrix.

Arms, identical folds, each with the hier2 head (lambda noise sharing, gamma class sharing):
    one_K0        the unified object with the constant row only     == unified.py, control
    one_KT        the same object with the full temporal basis      == the fix
    blocks        the separate-then-concatenate pipeline
    one_KT+blocks both

Measured (ph11, 15 folds): one_K0 0.171, one_KT 0.231, blocks 0.221, one_KT+blocks 0.255.
The K_time dose-response is an inverted U peaking at 7 (0.171 / 0.191 / 0.214 / 0.231 / 0.222
/ 0.206 for K = 0 / 1 / 3 / 7 / 15 / 23), and 7 half-cycles over a 1.01 s window puts the
highest retained component near 3.5 Hz, the N1/P2 timescale. Matched-row-count controls for
the temporal rows live in arch.py --study phi.

A scalp-Laplacian reservoir was also tested here and has been removed along with the rest of
the reservoir line. What it measured, for the record: geometry in the recurrent operator beat
a matched random reservoir by +0.024 (15/15 folds, p<1e-4), but the reservoir added nothing to
a plain delay embedding (res_lap+one_KT 0.217 against one_KT 0.236, p<1e-4), so the gain came
from spatial structure rather than recurrent dynamics and there are more direct ways to use
spatial structure.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, basis_modes, head, laplacian_modes, lda_persubj, lda_pooled,
                    load_pooled, pca_blocks, tangent, unified_spd)

ARMS = ["one_K0", "one_KT", "blocks", "one_KT+blocks"]


def _fold(args):
    (X, V, y, sid, tr, te, kind, d, lo, p, tau, K_time, phi, K_win,
     n_cls, n_sub, inner, seed, arms) = args
    U = basis_modes(kind, X, tr, V, d, lo)
    Z = np.einsum("ck,nct->nkt", U, X)

    reps = {}
    if "one_K0" in arms:
        reps["one_K0"] = tangent(unified_spd(Z, tr, p, tau, 0, phi), tr, sid)
    if arms & {"one_KT", "one_KT+blocks"}:
        reps["one_KT"] = tangent(unified_spd(Z, tr, p, tau, K_time, phi), tr, sid)
    if arms & {"blocks", "one_KT+blocks"}:
        reps["blocks"] = pca_blocks(X, tr, d, K_win, p, tau, sid=sid, subject_center=True)
    if "one_KT+blocks" in arms:
        reps["one_KT+blocks"] = np.hstack([reps["one_KT"], reps["blocks"]])
    reps = {k: v for k, v in reps.items() if k in arms}

    out = {}
    for nm, F in reps.items():
        acc, lgk = head(F, y, sid, tr, te, n_cls, n_sub, inner, seed)
        out[nm] = acc
        out[nm + "_lgk"] = lgk
        out[nm + "_persubjLDA"] = lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
        out[nm + "_pooledLDA"] = lda_pooled(F, y, tr, te)
    out["dim"] = {k: v.shape[1] for k, v in reps.items()}
    return out


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    V, _ = laplacian_modes(pos)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    arms = [x for x in ARMS if x in a.arms.split(",")] if a.arms else ARMS
    print("task=%s trials=%d subjects=%d classes=%d | basis=%s d=%d lo=%d p=%d tau=%d "
          "K_time=%d phi=%.2f | object side %d" % (
              a.task, len(X), n_sub, n_cls, a.basis, a.d, a.lo, a.p, a.tau, a.K_time,
              a.phi, a.p * a.d + a.K_time + 1), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, V, y, sid, tr, te, a.basis, a.d, a.lo, a.p, a.tau,
                         a.K_time, a.phi, a.windows, n_cls, n_sub, a.inner, sd, set(arms)))

    NAMES = arms + [x + s for x in arms for s in ("_persubjLDA", "_pooledLDA")]
    R = {k: [] for k in NAMES}
    LGK = {k: [] for k in arms}
    dims = None
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k in NAMES:
                R[k].append(out[k])
            for k in arms:
                LGK[k].append(out[k + "_lgk"])
            dims = out["dim"]
            print("  fold  " + "  ".join("%s %.3f" % (k, out[k]) for k in arms), flush=True)

    print()
    print("  tangent dims: " + "  ".join("%s %d" % (k, dims[k]) for k in arms))
    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    print("%-16s %8s %8s %8s | %5s %5s %5s" % (
        "representation", "hier2", "persubj", "pooled", "lam*", "gam*", "kap*"))
    for k in arms:
        lg = np.mean(LGK[k], axis=0)
        print("%-16s %8.3f %8.3f %8.3f | %5.2f %5.2f %5.2f" % (
            k, np.mean(R[k]), np.mean(R[k + "_persubjLDA"]), np.mean(R[k + "_pooledLDA"]),
            lg[0], lg[1], lg[2]))
    print()
    ref = np.array(R[a.ref if a.ref in arms else arms[0]], float)
    for k in sorted(arms, key=lambda k: -np.mean(R[k])):
        v = np.array(R[k], float)
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-16s %.3f  vs-chance p=%.7f  vs-%s %+.3f (p=%.4f) %s" % (
            k, v.mean(), p0, a.ref[:9], v.mean() - ref.mean(), pb,
            "**BEATS**" if pb < 0.05 and v.mean() > ref.mean() else ""))
    print()
    print("  one_K0 vs one_KT -- value of putting time order inside the single object")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--basis", default="pca", choices=["pca", "lap"])
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--lo", type=int, default=20)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K_time", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--arms", default="")
    ap.add_argument("--ref", default="blocks")
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
