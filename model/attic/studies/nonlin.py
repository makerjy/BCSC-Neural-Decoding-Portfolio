
"""
Does the object use nonlinear delay dynamics, or only the linear cross-lag correlation?

The 36x36 object in final_model.py is entirely second order. Every entry is a time-average of
a product of two rows, and the rows are Z(t), Z(t-tau) and fixed temporal functions. The
Z(t) x Z(t-tau) block is the *linear* lagged correlation -- the second-order signature a
linear DDE leaves behind -- but nothing in the matrix can represent a nonlinear delay
coupling. In DDA terms the a1 term is visible and the a3 x(t-tau)^2 term is invisible.

Putting it in is one line: make products of delayed signals into rows of their own, so the
covariance picks up third-order moments.

    A(t) = [ Z(t) ; Z(t-tau) ; Q(t) ; phi_0..phi_K ]

    Q = "self"   Z_i(t) * Z_i(t-tau)        quadratic delay coupling, DDA a3 analogue
        "sq"     Z_i(t-tau)^2               delayed energy
        "both"   both, on the top m/2 filters each

What this repo already measured, all on the older pipeline and none of it inside this object:
    DDA a3 median 0.00004 against a1 0.033, R^2 = 0.009
    neuron type: tanh / ReLU / sine / quadratic / LIF / Izhikevich all tie
    reservoir spectral radius: rho=0.5 gives 0.596, rho=1.3 gives 0.600
    NG-RC (delay polynomial features, separate vector not rows) 0.121, lost badly
So the prior is that the nonlinear content is absent. This measures it in the current object.

Controls, all at matched row count and matched row RMS:

    real            Q from the trial's own delayed signal
    shuffled        delayed copy taken from a different trial of the SAME SUBJECT, class
                    ignored. Same marginals, no labels used, within-trial coupling destroyed.
    shifted         delayed copy circularly shifted by T/2 inside the same trial -- keeps the
                    trial's own envelope, breaks the lag relationship. Label-free.
    linear          m extra *linear* rows at a second delay instead of products -- capacity
                    control, answers "is it just more rows"
    none            the current object
    shuffled_LEAKY  partner matched on class as well. Preserving the row's class-conditional
                    statistics requires knowing the test trial's label, so this arm leaks. It
                    is printed to show the size of that leak and must not be counted.

    real vs shuffled and real vs shifted are the tests.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, lda_persubj, load_pooled, phi_rows, snr_filter, tangent)

def partner_index(y, sid, rng, use_class=False):
    """Another trial from the same subject for each trial.

    use_class=True matches the partner's class as well, which preserves the row's
    class-conditional statistics but REQUIRES KNOWING THE TEST TRIAL'S LABEL to pick its
    partner -- that is label leakage, and the arm built from it is reported only to be
    discarded. use_class=False pairs within subject only and uses no labels.
    """
    part = np.arange(len(y))
    for s in np.unique(sid):
        for c in (np.unique(y) if use_class else [None]):
            i = np.where((sid == s) if c is None else ((sid == s) & (y == c)))[0]
            if len(i) < 2:
                continue
            perm = rng.permutation(len(i))
            for _ in range(8):
                if not np.any(perm == np.arange(len(i))):
                    break
                perm = rng.permutation(len(i))
            part[i] = i[perm]
    return part


def build(Z, tr, p, tau, K, m, mode, kind, phi_scale, part, shrink=0.08, rng=None):
    """Rows: Z(t) | Z(t-tau) | Q (m rows, per `kind`) | phi. Q is empty when kind='none'."""
    T0 = Z.shape[2]
    span = tau
    cur = Z[:, :, :T0 - span]
    lag = Z[:, :, span:]
    T = cur.shape[2]
    rows = [cur, lag]

    if kind != "none":
        if kind == "linear":                       # capacity control: a second linear delay
            span2 = min(2 * tau, T0 - T - 1)
            rows.append(Z[:, :m, span2:span2 + T])
        else:
            if kind == "real":
                lag_src = lag
            elif kind == "shuffled":
                lag_src = lag[part]                # same subject, different trial, no labels
            elif kind == "shuffled_LEAKY":
                lag_src = lag[part]                # caller passes the class-matched partners
            elif kind == "shifted":
                lag_src = np.roll(lag, T // 2, axis=2)
            else:
                raise ValueError(kind)
            half = m // 2
            q = []
            if mode in ("self", "both"):
                q.append(cur[:, :half if mode == "both" else m] *
                         lag_src[:, :half if mode == "both" else m])
            if mode in ("sq", "both"):
                k = half if mode == "both" else m
                q.append(lag_src[:, :k] ** 2)
            Q = np.concatenate(q, 1)
            Q = Q - Q.mean(2, keepdims=True)       # products have a large DC term
            rows.append(Q)

    A = np.concatenate(rows, 1)
    sc = np.sqrt((A[tr] ** 2).mean(axis=(0, 2), keepdims=True))
    A = A / np.maximum(sc, 1e-12)                  # match row RMS across all variants
    P = phi_rows("dct_low", T, K, 0) * phi_scale
    A = np.concatenate([A, np.broadcast_to(P, (len(A),) + P.shape)], 1)
    k = A.shape[1]
    C = np.einsum("nit,njt->nij", A, A) / T
    tc = np.trace(C, axis1=1, axis2=2)[:, None, None] / k
    return (1 - shrink) * C + shrink * tc * np.eye(k)


def _fold(args):
    (X, y, sid, tr, te, n_cls, d, p, tau, K, m, mode, phi_scale, eps, seed) = args
    rng = np.random.default_rng(seed)
    W = snr_filter(X, y, sid, tr, n_cls, d, eps, "noise")
    Z = np.einsum("ck,nct->nkt", W, X)
    part = partner_index(y, sid, rng, use_class=False)
    part_leaky = partner_index(y, sid, rng, use_class=True)
    out = {}
    for kind in KINDS:
        C = build(Z, tr, p, tau, K, m, mode, kind, phi_scale,
                  part_leaky if kind == "shuffled_LEAKY" else part)
        F = tangent(C, tr, sid, "train")
        out[kind] = lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
        out[kind + "_dim"] = F.shape[1]
    return out


KINDS = ["real", "shuffled", "shifted", "linear", "none", "shuffled_LEAKY"]


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    print("task=%s trials=%d classes=%d | d=%d tau=%d K=%d  quadratic rows m=%d mode=%s "
          "| side %d vs %d" % (a.task, len(X), n_cls, a.d, a.tau, a.K, a.m, a.mode,
                               2 * a.d + a.m + a.K + 1, 2 * a.d + a.K + 1), flush=True)

    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, y, sid, tr, te, n_cls, a.d, a.p, a.tau, a.K, a.m, a.mode,
                         a.phi, a.eps, sd))

    R = {k: [] for k in KINDS}
    dims = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for o in ex.map(_fold, jobs):
            for k in KINDS:
                R[k].append(o[k]); dims[k] = o[k + "_dim"]
            print("  fold  " + "  ".join("%s %.3f" % (k, o[k]) for k in KINDS), flush=True)

    print()
    print("=== %s (chance %.3f)  nonlinear delay rows ===" % (a.task, chance))
    print("%-10s %8s %6s" % ("rows", "persubj", "dim"))
    for k in KINDS:
        print("%-10s %8.3f %6d" % (k, np.mean(R[k]), dims[k]))
    print()
    ref = np.array(R["real"], float)
    for k in sorted(KINDS, key=lambda k: -np.mean(R[k])):
        v = np.array(R[k], float)
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-10s %.3f  vs-chance p=%.7f  vs-real %+.3f (p=%.4f)" % (
            k, v.mean(), p0, v.mean() - ref.mean(), pb))
    print()
    print("  real vs shuffled and real vs shifted are the tests: same row count, same RMS,")
    print("  no labels used, within-trial temporal coupling destroyed. A gap means a")
    print("  nonlinear delay coupling is being used; no gap means there is none.")
    print("  shuffled_LEAKY picks the partner by class, so it needs the test label -- it is")
    print("  printed to show what leakage looks like here, and must not be counted.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--mode", default="both", choices=["self", "sq", "both"])
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
