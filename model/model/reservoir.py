
"""
The connectionist arm without backpropagation.

`spdnet.py` asked whether a *learned* representation beats a designed one and answered no.
That is not the same question as whether an *emergent* one does. Connectionism does not
require gradient descent: in reservoir computing the recurrent weights are random and frozen,
nothing about the representation is designed or trained, and whatever capability appears comes
out of the collective dynamics of many simple units. Only the readout is fitted. This file is
that test, and it contains no backward pass anywhere.

Why it needs redoing rather than citing. The session already concluded that the reservoir
lineage adds nothing, but that conclusion rests on the weakest measurement in the repository:

    one setting only    rho=0.90, leak=0.30, gain=1.00, tanh -- while the delay embedding it
                        was compared against got a full tau sweep. Asymmetric search.
    wrong front end     basis=pca d=14 lo=20, not the production snr_filter.
    readout drift       the quoted -0.019 is the hier2 column; under the per-subject readout
                        the production model uses, the gap is -0.003 and pooled it is +0.001.
    unreproducible      the code was deleted, so it cannot be rerun.

The structure here removes all four. The reservoir is inserted *before* an otherwise identical
pipeline, so `designed` is not "the model without a reservoir" -- it is the identity reservoir:

    X (61 x T)
      -> r(t) = (1-a) r(t-1) + a f(W r(t-1) + W_in x(t)),  r(-1) = 0,  W and W_in frozen
      -> snr_filter on the states, N -> d                     supervised, training fold only
      -> unified_spd -> log-Euclidean tangent -> lda_persubj   byte-identical to final_model

With a=1, W=0, f=identity and W_in=I the recursion collapses to r(t) = x(t), so that arm must
reproduce `final_model` exactly. `--selftest` checks it does, which is what makes any other
number attributable to the dynamics rather than to a difference in plumbing.

The bias is deliberately set in the hypothesis's favour. `--study sweep` searches rho, leak,
nonlinearity and reservoir size, then the best cell is rerun at 15 folds. That best cell is
optimistically biased, because it was chosen on folds that overlap the ones it is scored on.
The argument only works in one direction, and it is the direction that matters here: if the
reservoir still loses after being handed its best setting and the selection bias too, it
loses. If it wins, the win has to be re-measured honestly before anyone believes it.

Arms
    designed        identity reservoir -- the reference, equals final_model
    res_rand        random W at the given spectral radius
    res_lap         W = -L/rho(L) from the montage graph Laplacian, so spatial frequency sets
                    the time constant of each mode. Requires N = 61, one node per electrode.
                    res_lap vs res_rand at matched rho/leak/nonlinearity is the one reservoir
                    result that was solidly measured before (+0.0235, 15/15) and it is
                    reproduced here rather than cited.
    res+designed    concatenated tangents -- does the reservoir *add* to the delay embedding,
                    which is the question the deleted experiment was actually asking

RESULTS

Sweep, 26 cells + designed, 5 folds (logs/reservoir_sweep_ph11.log). designed = 0.252.

    rho \\ nl      tanh          relu          linear
    0.3        0.233-0.237   0.216-0.222   0.236-0.245
    0.9        0.227-0.230   0.211-0.213   0.207-0.223
    1.3        0.197-0.209   0.175-0.181   0.096-0.152, one diverged

Every cell loses. What matters is not that, but the shape: accuracy falls monotonically with
the spectral radius, the nonlinearities do not help (linear >= tanh > relu), and the best cell
is rho=0.3, leak=1.0, linear -- weak recurrence, no integration memory, no nonlinearity. The
optimum sits in the corner of the space closest to having no reservoir at all, and even there
it loses to having none (0.245 vs 0.252). Size goes the same way: N = 61 -> 0.245,
122 -> 0.214, 244 -> 0.218, so the high-dimensional expansion that is supposed to be the point
of the method makes it worse.

15 folds at the best cell (logs/reservoir_ph11.log), rho=0.3 leak=1.0 linear N=61:

    res_lap+designed  0.249   +0.002  CI [-0.002,+0.006]  p=0.31
    res_rand+designed 0.248   +0.001  CI [-0.006,+0.008]  p=0.75
    designed          0.248        --
    res_lap           0.244   -0.003  CI [-0.008,+0.002]  p=0.21
    res_rand          0.241   -0.007  CI [-0.013,-0.001]  p=0.021

The concatenation intervals are +/-0.006, tight enough to say the reservoir adds nothing
rather than that we failed to detect it adding something. And the best cell was chosen on
folds that overlap these, so it is scored with the selection bias left in its favour -- the
deck is stacked for the hypothesis and it still loses.

The geometry claim reproduces and is now better than it was: at the old cell (rho=0.9,
leak=0.3, tanh) res_lap 0.251 vs res_rand 0.230, and res_lap ties `designed` outright
(0.251 vs 0.252) rather than losing to it as it did on the pca front end. So "putting the
montage into the recurrent operator beats matched randomness" survives; "the reservoir is
worth having" does not. Note the gap is cell-dependent -- +0.021 at rho=0.9/tanh but +0.003 at
rho=0.3/linear -- which is what it should be, since at rho=0.3 with leak=1.0 the recurrence
barely acts and the structure of W has almost nothing to act through.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import itertools

import numpy as np

from sdcore import (CHANCE, DATA, folds_for, lda_persubj, lda_pooled, laplacian_modes,
                    load_pooled, montage_laplacian, report, snr_filter, tangent, unified_spd)

NL = {"tanh": np.tanh,
      "relu": lambda v: np.maximum(v, 0.0),
      "linear": lambda v: v,
      "sine": np.sin}


# ---------------------------------------------------------------- the reservoir

def reservoir_W(kind, N, rho, pos=None, seed=0):
    """Frozen recurrent operator, scaled to spectral radius rho.

    kind="rand"  iid gaussian -- structure comes from nothing but the scaling
    kind="lap"   -L / rho(L) from the montage graph Laplacian. In the linear regime the
                 eigenmodes are the Laplacian's and mode k has pole (1-a) - a*rho*lambda_k, so
                 spatial frequency literally sets the time constant. This is the version with
                 a mechanism behind it, and `rand` is its matched control.
    """
    if kind == "lap":
        if pos is None or N != len(pos):
            raise ValueError("res_lap needs one node per electrode (N=%d)" % len(pos))
        W = -montage_laplacian(pos)
    elif kind == "rand":
        W = np.random.default_rng(seed).standard_normal((N, N)) / np.sqrt(N)
    else:
        raise ValueError(kind)
    r = np.abs(np.linalg.eigvals(W)).max()
    return W * (rho / r) if r > 1e-12 else W


def reservoir_run(X, W, W_in, leak, nl):
    """r(t) = (1-a) r(t-1) + a f(W r(t-1) + W_in x(t)),  r(-1) = 0.

    Written so that a=1, W=0, f=identity, W_in=I gives back x(t) with no shift -- an update
    that responded to x(t-1) instead would make the identity arm a one-sample lag of the
    designed one and quietly break the reference.
    """
    n, _, T = X.shape
    N = W.shape[0]
    f = NL[nl]
    R = np.empty((n, N, T), dtype=np.float32)
    r = np.zeros((n, N), dtype=np.float32)
    U = np.einsum("kc,nct->nkt", W_in, X).astype(np.float32)   # drive, precomputed
    Wt = W.T.astype(np.float32)
    for t in range(T):
        r = (1.0 - leak) * r + leak * f(r @ Wt + U[:, :, t])
        R[:, :, t] = r
    return R


def states(X, pos, cfg):
    """The frozen expansion. Uses no labels and no fold information, so it is computed once
    per configuration and shared across folds."""
    if cfg["kind"] == "identity":
        return X
    n_ch = X.shape[1]
    N = cfg["N"]
    W = reservoir_W(cfg["kind"], N, cfg["rho"], pos, cfg["seed"])
    if N == n_ch:
        W_in = np.eye(n_ch)
    else:
        W_in = np.random.default_rng(cfg["seed"] + 7).standard_normal((N, n_ch)) / np.sqrt(n_ch)
    return reservoir_run(X, W, cfg["gain"] * W_in, cfg["leak"], cfg["nl"])


# ---------------------------------------------------------------- scoring

def feats(R, y, sid, tr, n_cls, a, loso):
    W = snr_filter(R, y, sid, tr, n_cls, a.d, a.eps, a.den)
    Z = np.einsum("ck,nct->nkt", W, R)
    C = unified_spd(Z, tr, a.p, a.tau, a.K, a.phi)
    return tangent(C, tr, sid, "unlabeled" if loso else "train")


def score(F, y, sid, tr, te, n_cls, loso):
    return lda_pooled(F, y, tr, te) if loso \
        else lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)


def cfg_of(kind, a, rho=None, leak=None, nl=None, N=None):
    return {"kind": kind, "rho": rho if rho is not None else a.rho,
            "leak": leak if leak is not None else a.leak,
            "nl": nl if nl is not None else a.nl,
            "N": N if N is not None else a.N,
            "gain": a.gain, "seed": a.seed}


def run_arms(X, y, sid, pos, folds, n_cls, a, arms, loso, concat_with=None):
    """arms maps name -> config. Returns name -> per-fold accuracy.

    Arms are the outer loop on purpose. The states array is (trials x N x T) float32 -- 208 MB
    at N=61 and 832 MB at N=244 -- so a sweep that cached every configuration at once would
    need tens of gigabytes. Only one lives at a time.
    """
    res, base_feats = {}, None
    if concat_with is not None:
        Rb = states(X, pos, arms[concat_with])
        base_feats = [feats(Rb, y, sid, tr, n_cls, a, loso) for tr, _ in folds]
        del Rb

    for name, c in arms.items():
        R = states(X, pos, c)
        if not np.isfinite(R).all():
            # a linear reservoir above rho=1 has no bounding nonlinearity and grows without
            # limit. That is the dynamics, not a bug, so it is recorded as a diverged arm
            # rather than allowed to stop the sweep.
            res[name] = [np.nan] * len(folds)
            print("  %-18s diverged (rho=%.1f leak=%.1f %s)"
                  % (name, c["rho"], c["leak"], c["nl"]), flush=True)
            del R
            continue
        for fi, (tr, te) in enumerate(folds):
            F = feats(R, y, sid, tr, n_cls, a, loso)
            res.setdefault(name, []).append(score(F, y, sid, tr, te, n_cls, loso))
            if base_feats is not None and name != concat_with:
                tag = name + "+" + concat_with
                res.setdefault(tag, []).append(
                    score(np.hstack([F, base_feats[fi]]), y, sid, tr, te, n_cls, loso))
        del R
        msg = "  %-18s %.3f" % (name, np.mean(res[name]))
        tag = name + "+" + concat_with if concat_with and name != concat_with else None
        if tag:
            msg += "   %s %.3f" % (tag, np.mean(res[tag]))
        print(msg + "   (%d folds)" % len(folds), flush=True)
    return res


# ---------------------------------------------------------------- self test

def selftest(a):
    """The identity reservoir must reproduce the designed pipeline exactly. If it does not,
    every other arm is measuring plumbing rather than dynamics."""
    ok = True
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 61, 40)).astype(np.float32)

    R = reservoir_run(X, np.zeros((61, 61)), np.eye(61), 1.0, "linear")
    d1 = float(np.abs(R - X).max())
    print("  identity recursion   max|R - X| = %.3e" % d1)
    ok &= d1 < 1e-6

    lin = reservoir_run(X, np.zeros((61, 61)), np.eye(61), 0.5, "linear")
    ref = np.zeros_like(X)
    r = np.zeros((6, 61))
    for t in range(X.shape[2]):
        r = 0.5 * r + 0.5 * X[:, :, t]
        ref[:, :, t] = r
    d2 = float(np.abs(lin - ref).max())
    print("  leaky integrator     max|R - reference| = %.3e" % d2)
    ok &= d2 < 1e-5

    for kind, N in [("rand", 40), ("lap", 61)]:
        pos = rng.standard_normal((61, 3)) if kind == "lap" else None
        W = reservoir_W(kind, N, 0.9, pos)
        r = float(np.abs(np.linalg.eigvals(W)).max())
        print("  spectral radius %-4s requested 0.900  achieved %.6f" % (kind, r))
        ok &= abs(r - 0.9) < 1e-6

    Rs = reservoir_run(X, reservoir_W("rand", 61, 0.9), np.eye(61), 0.3, "tanh")
    print("  echo state (rho 0.9) max|r| = %.3f   finite %s"
          % (np.abs(Rs).max(), bool(np.isfinite(Rs).all())))
    ok &= bool(np.isfinite(Rs).all()) and np.abs(Rs).max() < 100

    print("\n  SELFTEST %s" % ("PASSED" if ok else "FAILED"))
    return ok


# ---------------------------------------------------------------- main

def main(a):
    if a.selftest:
        raise SystemExit(0 if selftest(a) else 1)

    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    n_cls = len(np.unique(y)); chance = CHANCE[a.task]
    loso = a.split == "loso"
    folds = folds_for(a.split, X, y, tok, sid, a.seeds)
    print("task=%s trials=%d subjects=%d classes=%d | study=%s split=%s folds=%d | "
          "d=%d p=%d tau=%d K=%d" % (a.task, len(X), n_sub, n_cls, a.study, a.split,
                                     len(folds), a.d, a.p, a.tau, a.K), flush=True)

    if a.study == "sweep":
        # every axis the deleted experiment held fixed, on the production front end
        print("  sweeping rho x leak x nonlinearity at N=%d, plus N at the default cell"
              % a.N, flush=True)
        arms = {"designed": cfg_of("identity", a)}
        for rho, leak, nl in itertools.product(a.rhos, a.leaks, a.nls):
            arms["r%.1f_l%.1f_%s" % (rho, leak, nl)] = cfg_of("rand", a, rho, leak, nl)
        for N in a.Ns:
            if N != a.N:
                arms["N%d" % N] = cfg_of("rand", a, N=N)
        res = run_arms(X, y, sid, pos, folds, n_cls, a, arms, loso)
    elif a.study == "geom":
        # the one reservoir claim that was solidly measured: geometry beats matched randomness
        arms = {"designed": cfg_of("identity", a),
                "res_lap": cfg_of("lap", a, N=X.shape[1]),
                "res_rand": cfg_of("rand", a, N=X.shape[1])}
        res = run_arms(X, y, sid, pos, folds, n_cls, a, arms, loso)
    else:
        arms = {"designed": cfg_of("identity", a),
                "res_rand": cfg_of("rand", a),
                "res_lap": cfg_of("lap", a, N=X.shape[1])}
        res = run_arms(X, y, sid, pos, folds, n_cls, a, arms, loso, concat_with="designed")

    print()
    print("=== %s (chance %.3f)  study=%s  %d folds | rho=%.2f leak=%.2f nl=%s N=%d ==="
          % (a.task, chance, a.study, len(folds), a.rho, a.leak, a.nl, a.N))
    report(res, chance, ref="designed", label="per-subject balanced accuracy", width=22)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--study", default="main", choices=["main", "sweep", "geom"])
    ap.add_argument("--split", default="trial", choices=["trial", "token", "time", "loso"])
    ap.add_argument("--selftest", action="store_true")
    # reservoir
    ap.add_argument("--N", type=int, default=61)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--nl", default="tanh", choices=list(NL))
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.3, 0.9, 1.3])
    ap.add_argument("--leaks", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    ap.add_argument("--nls", nargs="+", default=["tanh", "relu", "linear"])
    ap.add_argument("--Ns", nargs="+", type=int, default=[61, 122, 244])
    # representation, held at final_model.py's values
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--den", default="noise", choices=["eye", "noise"])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=1)
    main(ap.parse_args())
