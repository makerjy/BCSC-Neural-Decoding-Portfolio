
"""
The three-stage architecture as specified, plus the controls that decide whether each stage
works for the reason claimed.

    input -> FILTER -> BASE MODEL -> readout

Everything before this script drifted away from that. The best number of the session,
band_one+blocks 0.313, is a concatenation of two representations bolted together because the
measurement said it helped; it is not a design. The designed model is band_one, 0.274, and
its FILTER stage was never built -- electrode coordinates were used as a *basis* inside the
base model, and subject handling was pushed downstream into the readout's lambda and gamma.

STAGE 1  FILTER -- separate subject variance from phoneme variance, using the montage.

    per (subject, class) mean waveform M_sc (61 x T), from training trials only
      S_phon = sum_s sum_c (M_sc - M_s)(M_sc - M_s)^T     within-subject, between-class
      S_subj = sum_s      (M_s  - M  )(M_s  - M  )^T      between-subject

    W = top-d generalised eigenvectors of  S_phon  against  nuis*S_subj + alpha*L + eps*I

    all three denominator terms trace-normalised. L is the montage graph Laplacian, so
    alpha penalises spatially rough filters: a real cortical source projects smoothly through
    the skull, and an unconstrained 61x61 filter can permute distant electrodes, which is
    physically impossible and is what the earlier Procrustes attempt did (-0.008).

    2x2 ablation, because each term has to earn its place:
      nuis=1 alpha=1   the design
      nuis=1 alpha=0   subject suppression without the geometry prior
      nuis=0 alpha=1   geometry prior without subject suppression
      nuis=0 alpha=0   plain class-discriminant filter
    plus lap_band / pca / rand as fixed-basis and capacity controls.

STAGE 2  BASE MODEL -- one SPD object per trial, space and time melted in it.

    A(t) = [ Z(t) ; Z(t-tau) ; phi_0(t) ... phi_K(t) ],   C = A A^T / T

    The claim is that the phi rows put *time order* into an object that otherwise averages
    over t. That claim adds K+1 rows, so it also adds dimensions, and no control so far
    separates the two. These do, at matched row count:

      dct_low   k = 0..K            the claim: smooth, low frequency, ordered
      dct_high  k = H..H+K          same smoothness class, wrong timescale
      shuffled  dct_low with its time samples permuted by one fixed permutation
                -- same values, same marginal distribution, no coherent time structure
      gauss     iid rows at matched RMS -- no structure at all
      none      no rows, centred covariance (= the object before the fix)

    If dct_low does not beat all three matched controls, the mechanism claim is wrong and the
    gain was dimensional.

STAGE 3  READOUT -- one Gaussian discriminant, one dense Sigma^-1, with the two subject
    shrinkages measured earlier (lambda on the noise geometry, gamma on the class means).
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, disc_filter, laplacian_modes, lda_persubj, lda_pooled, load_pooled,
                    montage_laplacian, tangent, unified_spd)

# ------------------------------------------------------------------ stage 1

# ------------------------------------------------------------------ stage 2

# ------------------------------------------------------------------ fold

def _fold(args):
    (X, V, L, y, sid, tr, te, study, d, lo, p, tau, K, high, phi_scale,
     eps, n_cls, n_sub, seed) = args
    out = {}
    if study == "filter":
        cfgs = [("design_nuis+lap", "disc", 1.0, 1.0),
                ("nuis_only", "disc", 1.0, 0.0),
                ("lap_only", "disc", 0.0, 1.0),
                ("plain_disc", "disc", 0.0, 0.0),
                ("lap_band", "lap_band", 0.0, 0.0),
                ("pca", "pca", 0.0, 0.0),
                ("rand", "rand", 0.0, 0.0)]
        for nm, kind, nu, al in cfgs:
            W = disc_filter(X, y, sid, tr, n_cls, V, L, d, lo, nu, al, eps, kind, seed)
            Z = np.einsum("ck,nct->nkt", W, X)
            F = tangent(unified_spd(Z, tr, p, tau, K, phi_scale, "dct_low", high,
                                    seed=seed), tr, sid)
            out[nm + "_persubj"] = lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
            out[nm + "_pooled"] = lda_pooled(F, y, tr, te)
    else:
        W = disc_filter(X, y, sid, tr, n_cls, V, L, d, lo, 1.0, 1.0, eps, "disc", seed)
        Z = np.einsum("ck,nct->nkt", W, X)
        for nm in ["dct_low", "dct_high", "shuffled", "gauss", "none"]:
            F = tangent(unified_spd(Z, tr, p, tau, K, phi_scale, nm, high, seed=seed),
                        tr, sid)
            out[nm + "_persubj"] = lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
            out[nm + "_pooled"] = lda_pooled(F, y, tr, te)
            out[nm + "_dim"] = F.shape[1]
    return out


FILTER_ARMS = ["design_nuis+lap", "nuis_only", "lap_only", "plain_disc",
               "lap_band", "pca", "rand"]
PHI_ARMS = ["dct_low", "dct_high", "shuffled", "gauss", "none"]


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    V, _ = laplacian_modes(pos)
    L = montage_laplacian(pos)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    ARMS = FILTER_ARMS if a.study == "filter" else PHI_ARMS
    print("task=%s trials=%d subjects=%d classes=%d | study=%s d=%d p=%d tau=%d K=%d "
          "high=%d | object side %d" % (
              a.task, len(X), n_sub, n_cls, a.study, a.d, a.p, a.tau, a.K, a.high,
              a.p * a.d + a.K + 1), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, V, L, y, sid, tr, te, a.study, a.d, a.lo, a.p, a.tau,
                         a.K, a.high, a.phi, a.eps, n_cls, n_sub, sd))

    R = {k + s: [] for k in ARMS for s in ("_persubj", "_pooled")}
    dims = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for o in ex.map(_fold, jobs):
            for k in R:
                R[k].append(o[k])
            for k in ARMS:
                if k + "_dim" in o:
                    dims[k] = o[k + "_dim"]
            print("  fold  " + "  ".join("%s %.3f" % (k[:13], o[k + "_persubj"])
                                         for k in ARMS), flush=True)

    print()
    print("=== %s (chance %.3f)  study=%s ===" % (a.task, chance, a.study))
    print("%-18s %8s %8s %6s" % ("arm", "persubj", "pooled", "dim"))
    for k in ARMS:
        print("%-18s %8.3f %8.3f %6s" % (k, np.mean(R[k + "_persubj"]),
                                         np.mean(R[k + "_pooled"]), dims.get(k, "-")))
    print()
    ref = np.array(R[ARMS[0] + "_persubj"], float)
    for k in sorted(ARMS, key=lambda k: -np.mean(R[k + "_persubj"])):
        v = np.array(R[k + "_persubj"], float)
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-18s %.3f  vs-chance p=%.7f  vs-%s %+.3f (p=%.4f)" % (
            k, v.mean(), p0, ARMS[0][:9], v.mean() - ref.mean(), pb))
    if a.study == "phi":
        print()
        print("  dct_low must beat dct_high, shuffled and gauss -- all three have the same")
        print("  row count and scale, so any gap between them is time structure, not size.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--study", default="phi", choices=["filter", "phi"])
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--lo", type=int, default=20)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--high", type=int, default=40)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
