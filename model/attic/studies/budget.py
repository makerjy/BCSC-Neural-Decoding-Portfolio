
"""
The exchange rate: how many other people's trials are worth one of your own?

Every pooling experiment in this repo answers "pooled vs per-subject" at one fixed sample
size, which cannot distinguish "other subjects carry no information" from "other subjects
carry information but 16x208 trials of it is still worth less than 208 of your own". That
distinction decides whether the pooled architecture is worth building at all.

So hold out one subject's trials as the test set and sweep two axes independently:

    n_own    trials from the test subject that the model may train on   (0 .. all)
    others   trials from the other 15 subjects                          (none | all)

Three models on identical splits:
    own_only      subject's own trials, per-subject LDA
    pooled        own + all others, one classifier, no subject term
    hier          own + all others, shared covariance, class means shrunk toward the group
                  with g chosen by inner CV (contains both of the above)

Readouts:
    n_own = 0            what other people alone buy you (the LOSO number)
    n_own small          whether the group acts as a usable prior when your own data is thin
    n_own large          whether the group still adds once your own data is plentiful
    horizontal shift     how many own-trials the group is worth
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, blend, cell_means, gauss_predict, lda_pooled, load_subject,
                    pca_blocks, pick_gamma, shared_precision)

def _job(args):
    (X, y, sid, s, tr_own_all, te, tr_other, n_own, d, K, p, tau,
     n_cls, n_sub, grid, rng_seed) = args
    rng = np.random.default_rng(rng_seed)

    if n_own == 0:
        own = np.array([], int)
    elif n_own >= len(tr_own_all):
        own = tr_own_all
    else:                                   # class-stratified subsample of own trials
        take = []
        per = max(1, n_own // n_cls)
        for c in range(n_cls):
            ic = tr_own_all[y[tr_own_all] == c].copy()
            rng.shuffle(ic)
            take.append(ic[:per])
        own = np.concatenate(take)

    tr = np.concatenate([own, tr_other]) if len(tr_other) else own
    if len(own) == 0 and len(tr_other) == 0:
        return None
    fit = np.sort(np.concatenate([tr, te]))          # feature fit uses train trials only
    trf = np.array([np.searchsorted(fit, i) for i in np.sort(tr)])
    tef = np.array([np.searchsorted(fit, i) for i in np.sort(te)])
    Z = pca_blocks(X[fit], trf, d, K, p, tau, sid=sid[fit], subject_center=True)
    yf, sf = y[fit], sid[fit]

    out = {"n_own": n_own, "others": int(len(tr_other) > 0), "subj": int(s)}
    if len(own) >= 2 * n_cls and len(tr_other) == 0:
        out["own_only"] = lda_pooled(Z, yf, trf, tef)
    elif len(tr_other):
        out["own_only"] = np.nan
    else:
        out["own_only"] = np.nan

    if len(np.unique(yf[trf])) < n_cls:
        return out
    out["pooled"] = lda_pooled(Z, yf, trf, tef)
    M, N, Mp = cell_means(Z, yf, sf, trf, n_cls, n_sub)
    P = shared_precision(Z, yf, sf, trf, M)
    if len(tr_other) and len(own) >= 2 * n_cls:
        g, _ = pick_gamma(Z, yf, sf, trf, n_cls, n_sub, grid, inner=3, seed=rng_seed)
    else:
        g = 1.0 if len(own) < 2 * n_cls else 0.0
    out["gamma"] = g
    out["hier"] = balanced_accuracy_score(
        yf[tef], gauss_predict(Z, tef, sf, blend(M, Mp, g), P))
    return out


def main(a):
    files = [f for f in sorted(glob.glob(os.path.join(a.data, "*.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    Xs, ys, sid = [], [], []
    for i, f in enumerate(files):
        Xi, yi, _ = load_subject(f, a.task)
        Xs.append(Xi); ys.append(yi); sid.append(np.full(len(yi), i))
    X = np.concatenate(Xs); y = np.concatenate(ys); sid = np.concatenate(sid)
    n_sub = len(files); n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    print("task=%s trials=%d subjects=%d classes=%d  own-trials/subject~%d" % (
        a.task, len(X), n_sub, n_cls, len(X) // n_sub), flush=True)

    grid = np.linspace(0, 1, 11)
    subs = list(range(n_sub))[:a.max_subjects]
    jobs = []
    for s in subs:
        own_all = np.where(sid == s)[0]
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=0)
        for tr_o, te_o in list(skf.split(own_all, y[own_all]))[:a.folds_used]:
            tr_own_all, te = own_all[tr_o], own_all[te_o]
            other = np.where(sid != s)[0]
            for n_own in a.n_own:
                if n_own > len(tr_own_all):
                    continue
                for use_other in (0, 1):
                    if n_own == 0 and not use_other:
                        continue
                    jobs.append((X, y, sid, s, tr_own_all, te,
                                 other if use_other else np.array([], int),
                                 n_own, a.d, a.windows, a.p, a.tau,
                                 n_cls, n_sub, grid, 1000 * s + n_own))
    print("jobs=%d" % len(jobs), flush=True)

    rec = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, o in enumerate(ex.map(_job, jobs)):
            if o:
                rec.append(o)
            if (i + 1) % 20 == 0:
                print("  %d/%d" % (i + 1, len(jobs)), flush=True)

    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    print("%8s | %-22s | %-30s" % ("n_own", "own trials only", "own + all 15 other subjects"))
    print("%8s | %10s %10s | %10s %10s %8s" % (
        "", "own_only", "hier", "pooled", "hier", "g*"))
    for n_own in a.n_own:
        A = [r for r in rec if r["n_own"] == n_own and not r["others"]]
        B = [r for r in rec if r["n_own"] == n_own and r["others"]]
        def m(rows, k):
            v = [r[k] for r in rows if k in r and np.isfinite(r.get(k, np.nan))]
            return np.mean(v) if v else np.nan
        print("%8d | %10.3f %10.3f | %10.3f %10.3f %8.2f" % (
            n_own, m(A, "own_only"), m(A, "hier"),
            m(B, "pooled"), m(B, "hier"), m(B, "gamma")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds_used", type=int, default=2)
    ap.add_argument("--max_subjects", type=int, default=8)
    ap.add_argument("--n_own", type=int, nargs="+", default=[0, 22, 44, 88, 176])
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
