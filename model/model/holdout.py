
"""
An untouched final test set, because every number in this session was selected on the folds
it was measured on.

The session compared roughly ten representations on the same cross-validation folds and then
headlined the best (`band_one+blocks` 0.305). No arm was picked per fold -- each is one fixed
representation -- but the decision about which to headline was made after seeing all of their
test scores, and so were smaller choices like the filter denominator (`noise` over `eye`) and
the temporal-row count. That is an optimistic bias, and this repo has been burned by its
sharper form before: `u2.log`'s `U_best+base` selected the best variant per fold, which
`unified2b` fixed by fixing the variant a priori (`base+U_FIXED`).

The fix at session scale is a set of trials that selection never touches.

    final   20% of trials, stratified by (subject, class), seed 12345 -- used exactly once
    dev     80% -- every fit and every choice happens here, by inner CV

Two numbers come out:
    per model      inner-CV estimate on dev, then one score on final
    honest pick    the model inner CV would have chosen, and what it then scored on final

The gap between "best on final" and "honest pick on final" is the selection bias, measured
rather than argued about.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

from sdcore import (CHANCE, DATA, basis_full, block_features, laplacian_modes, lda_persubj,
                    lda_pooled, load_pooled, snr_filter, tangent, unified_spd, window_blocks)

SEED_FINAL = 12345          # used for nothing else in this session


def features(name, X, y, sid, fit, V, wlap, n_cls, a):
    """Everything fitted uses `fit` only. `name` selects one of the three models."""
    if name == "final_model":
        W = snr_filter(X, y, sid, fit, n_cls, a.d, a.eps, "noise")
        Z = np.einsum("ck,nct->nkt", W, X)
        return tangent(unified_spd(Z, fit, a.p, a.tau, a.K, a.phi), fit, sid)
    if name != "best_concat":
        raise ValueError(name)                     # eegnet is scored separately, see below
    U = basis_full("lap", X, fit, V, wlap, 0.0, 0.5)
    edges = np.linspace(0, X.shape[1], a.bands + 1).astype(int)
    one, blk = [], []
    for b in range(a.bands):
        Zb = np.einsum("ck,nct->nkt", U[:, edges[b]:edges[b + 1]], X)
        one.append(tangent(unified_spd(Zb, fit, a.p, a.tau, a.K, a.phi), fit, sid))
        blk.append(block_features(window_blocks(Zb, fit, a.windows, a.p, a.tau), fit, sid))
    return np.hstack(one + blk)


def eegnet_score(X, y, sid, fit, test, n_cls, n_sub, a):
    """EEGNet on the same split. Imported lazily so torch is only needed when asked for."""
    import torch
    import eegnet as E
    Xn = X[:, None].astype(np.float32)
    yl, sl = y.astype(np.int64), sid.astype(np.int64)
    o = E.run_fold(Xn, yl, sl, fit, np.sort(test), n_cls, n_sub,
                   torch.device(a.device), a, 0)
    return o["pooled+ft_head"], 4011              # trainable weights, shared across subjects


MODELS = ["final_model", "best_concat", "eegnet"]


def score(name, X, y, sid, fit, test, V, wlap, n_cls, n_sub, a, readout):
    if name == "eegnet":
        return eegnet_score(X, y, sid, fit, test, n_cls, n_sub, a)
    F = features(name, X, y, sid, fit, V, wlap, n_cls, a)
    if readout == "pooled":
        return lda_pooled(F, y, fit, test), F.shape[1]
    return lda_persubj(F, y, sid, fit, test, n_cls, 3 * n_cls), F.shape[1]


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    V, wlap = laplacian_modes(pos)
    n_cls = len(np.unique(y)); chance = CHANCE[a.task]
    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])

    dev_i, fin_i = next(iter(StratifiedKFold(5, shuffle=True, random_state=SEED_FINAL)
                             .split(X, strat)))
    dev, final = np.sort(dev_i), np.sort(fin_i)
    print("task=%s | dev %d시행 / final %d시행 (seed %d, 이 세션에서 처음 쓰는 분할)"
          % (a.task, len(dev), len(final), SEED_FINAL), flush=True)
    print("  final은 선택에 한 번도 쓰이지 않고 모델당 정확히 한 번 채점된다", flush=True)
    print(flush=True)

    inner = list(StratifiedKFold(a.inner, shuffle=True, random_state=0)
                 .split(dev, strat[dev]))
    rows = []
    for name in MODELS:
        est = []
        for itr, ite in inner:
            s, _ = score(name, X, y, sid, dev[itr], dev[ite], V, wlap, n_cls, n_sub,
                         a, a.readout)
            est.append(s)
        fin, dim = score(name, X, y, sid, dev, final, V, wlap, n_cls, n_sub, a, a.readout)
        rows.append((name, dim, float(np.mean(est)), float(np.std(est)), fin))
        print("  %-16s dim %5d  dev(inner CV) %.3f ± %.3f  ->  final %.3f"
              % (name, dim, np.mean(est), np.std(est), fin), flush=True)

    print()
    print("=== %s (우연 %.3f)  readout=%s ===" % (a.task, chance, a.readout))
    print("%-16s %6s %16s %10s" % ("model", "dim", "dev (inner CV)", "final"))
    for nm, dim, m, sd, fin in rows:
        print("%-16s %6d %10.3f ±%.3f %10.3f" % (nm, dim, m, sd, fin))
    pick = max(rows, key=lambda r: r[2])
    best = max(rows, key=lambda r: r[4])
    print()
    print("  inner CV가 고를 모델      : %s  ->  final %.3f" % (pick[0], pick[4]))
    print("  final에서 실제 최고        : %s  ->  final %.3f" % (best[0], best[4]))
    print("  선택 편향 (실제최고 − 정직한선택) = %+.3f" % (best[4] - pick[4]))
    print()
    print("  '정직한 선택'이 test를 한 번도 보지 않고 얻을 수 있는 값이다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--readout", default="persubj", choices=["persubj", "pooled"])
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")        # eegnet only
    ap.add_argument("--kern", type=int, default=128)     # eegnet hyperparameters, kept at
    ap.add_argument("--drop", type=float, default=0.25)  # the published EEGNet-8,2 values
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ft_epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    main(ap.parse_args())
