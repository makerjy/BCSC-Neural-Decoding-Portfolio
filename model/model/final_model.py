
"""
The model reduced to the parts that passed a control.

Design as specified:  input -> filter -> base model -> readout.
Every component below is here because removing it, or replacing it with a matched-size
version that lacks the claimed mechanism, costs accuracy. Components that only earned an
accuracy bump without surviving their control are gone, and are listed so they stay gone.

STAGE 1  FILTER -- supervised discriminant spatial projection, 61 -> d
    S_phon = sum_s sum_c (M_sc - M_s)(M_sc - M_s)^T     within-subject, between-class
    W = top-d generalised eigenvectors of S_phon against (within-class noise + eps I)
    The denominator is the within-(subject, class) residual scatter, not the identity: a
    discriminant filter should maximise signal-to-noise, not signal. Theory picks this first
    and the measurement agrees (0.252 vs 0.235).
    kept because: beats the fixed Laplacian band by +0.016 (p=0.016) and a random
    orthonormal projection of the same rank by +0.018 (p=0.005).
    removed: between-subject nuisance suppression (contributes -0.002, p=0.73, and alone
    costs -0.025, p=0.0002) and the Laplacian smoothness penalty (only repairs the damage
    the nuisance term causes; without it, -0.005 ns). The reason nuisance suppression fails
    is measured: subjects' discriminant directions are near-orthogonal (cos 0.01 against a
    within-subject split-half of 0.45), so there is no subject subspace sitting beside a
    shared phoneme subspace -- projecting out between-subject directions deletes the very
    directions that carry each individual's phoneme signal.

STAGE 2  BASE MODEL -- one SPD object per trial, space and time interacting inside it
    A(t) = [ Z(t) ; Z(t-tau) ; phi_0(t) ... phi_K(t) ],   C = A A^T / T
    kept because: at matched row count, dct_low beats time-shuffled rows (-0.054), iid rows
    (-0.066) and high-frequency rows (-0.101), all p<1e-4, and also beats having no rows at
    all (-0.052). The gain is time structure at the right timescale, not dimensions.
    removed: the scalp-Laplacian reservoir and the band-concatenation of two representations
    (an ensemble, not a design). The reservoir case is two claims of unequal strength. Solid:
    it beats a matched random reservoir by +0.0235, 15/15 folds, 95% CI [+0.015, +0.032], so
    putting geometry into a recurrent operator does work. Weak: it does not add to the delay
    embedding -- but the often-quoted -0.019 is the hier2 readout column, and under the
    per-subject readout this model actually uses the gap is -0.003 (pooled +0.001). It was
    also measured at a single setting (rho=0.90, leak=0.30, gain=1.00, tanh) on a different
    front-end (basis=pca d=14 lo=20), while the delay embedding got a tau sweep, so the search
    is asymmetric. What is established is "adding it does not help here", not "it is worse".

STAGE 3  READOUT -- tangent-space Gaussian with a personal decision rule
    kept because: EEGNet with a frozen shared trunk and a per-subject refitted head beats
    per-subject networks by +0.044 (p=0.0001) while plain pooling does not (+0.009, ns), and
    the calibration curve crosses over at 22-40 own trials -- beyond that, other subjects
    cost accuracy.
    removed: the interpretation of gamma as class-mean sharing. Shrinking toward the group
    mean is indistinguishable from shrinking toward the origin (0.713 vs 0.716 on C/V), so
    it is regularisation.

Controls this script runs:
    --study perm    label permutation null for this exact configuration
    --study split   trial / token / time / loso on one model
    --study den     the one remaining open design choice in stage 1
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel


from sdcore import (DATA, folds_for, lda_persubj, lda_pooled, load_pooled, snr_filter, tangent,
                    unified_spd)
def run(X, y, sid, tr, te, n_cls, d, p, tau, K, phi_scale, den_kind, eps, loso,
        adapt="train", readout="persubj"):
    """readout="persubj" one classifier per subject, "pooled" one for everyone, "dim" the
    feature length. A held-out subject has no labelled trials, so LOSO forces pooled."""
    W = snr_filter(X, y, sid, tr, n_cls, d, eps, den_kind)
    Z = np.einsum("ck,nct->nkt", W, X)
    F = tangent(unified_spd(Z, tr, p, tau, K, phi_scale), tr, sid, adapt)
    if readout == "dim":
        return float(F.shape[1])
    if loso or readout == "pooled":
        return lda_pooled(F, y, tr, te)
    return lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)


def _job(args):
    (X, y, sid, tr, te, n_cls, d, p, tau, K, phi_scale, den_kind, eps, loso,
     perm_seed, adapt, readout) = args
    if perm_seed is not None:                             # shuffle labels within subject
        rng = np.random.default_rng(perm_seed)
        y = y.copy()
        for s in np.unique(sid):
            i = np.where(sid == s)[0]
            y[i] = rng.permutation(y[i])
    return run(X, y, sid, tr, te, n_cls, d, p, tau, K, phi_scale, den_kind, eps, loso,
               adapt, readout)


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    print("task=%s trials=%d subjects=%d classes=%d | study=%s d=%d p=%d tau=%d K=%d "
          "den=%s | object side %d" % (a.task, len(X), n_sub, n_cls, a.study, a.d, a.p,
                                       a.tau, a.K, a.den, a.p * a.d + a.K + 1), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])

    def folds(kind):
        return folds_for(kind, X, y, tok, sid, a.seeds)

    def base(kind, den_kind, perm_seed=None, adapt="train", readout="persubj"):
        return [(X, y, sid, tr, te, n_cls, a.d, a.p, a.tau, a.K, a.phi, den_kind,
                 a.eps, kind == "loso", perm_seed, adapt, readout)
                for tr, te in folds(kind)]

    jobs, tags = [], []
    if a.study == "perm":
        jobs += base("trial", a.den); tags += ["real"] * len(folds("trial"))
        for r in range(a.perms):
            jobs += base("trial", a.den, perm_seed=1000 + r)
            tags += ["perm%d" % r] * len(folds("trial"))
    elif a.study == "split":
        for k in ["trial", "token", "time", "loso"]:
            jobs += base(k, a.den); tags += [k] * len(folds(k))
    elif a.study == "readout":       # the gap: per-subject vs pooled on identical folds
        for r in ["persubj", "pooled"]:
            jobs += base("trial", a.den, readout=r)
            tags += [r] * len(folds("trial"))
    elif a.study == "den":
        for dk in ["eye", "noise"]:
            jobs += base("trial", dk); tags += [dk] * len(folds("trial"))
    else:                                   # unsupervised adaptation for an unseen subject
        for kind in ["loso", "trial"]:
            for ad in ["train", "unlabeled"]:
                jobs += base(kind, a.den, adapt=ad)
                tags += ["%s_%s" % (kind, ad)] * len(folds(kind))

    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for t, v in zip(tags, ex.map(_job, jobs)):
            res.setdefault(t, []).append(v)
            if len(res[t]) == 1:
                print("  %s started" % t, flush=True)

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
            print("  %-16s %.3f +- %.3f  (%d folds)  vs-chance p=%.7f" % (
                k, v.mean(), v.std(), len(v), p0))
        if a.study == "adapt":
            print()
            for kind in ["loso", "trial"]:
                A = np.array(res.get(kind + "_train", []), float)
                B = np.array(res.get(kind + "_unlabeled", []), float)
                if len(A) and len(A) == len(B):
                    _, pp = ttest_rel(B, A)
                    print("  %-6s unlabeled - train  %+.4f   folds up %d/%d   paired p=%.4f"
                          % (kind, (B - A).mean(), int((B > A).sum()), len(A), pp))
                    print("         per fold: " + " ".join("%+.3f" % d for d in (B - A)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--study", default="split",
                    choices=["perm", "split", "den", "adapt", "readout"])
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--den", default="noise", choices=["eye", "noise"])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--perms", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
