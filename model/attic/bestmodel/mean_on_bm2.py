"""Add the discarded window means to the best_model2 pipeline (npz2, 0-1.0s, ICA-cleaned).

pyriemann's Covariances centres each window over time, so the per-window per-channel mean
voltage never reaches the classifier. On the npz dataset restoring it gave place +0.027 and
11-phoneme +0.022 (both significant). best_model2 does not use it, so this tests whether the
same gain transfers to the cleaner data and longer window.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os

import numpy as np
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

import best_model2 as BM


def parts(Xtr, Xte, n_win, n_pca):
    pca = PCA(min(n_pca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pca.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    ed = np.linspace(0, A.shape[2], n_win + 1).astype(int)
    ta, tb, ma, mb = [], [], [], []
    for i in range(n_win):
        sa, sb = A[:, :, ed[i]:ed[i+1]], B[:, :, ed[i]:ed[i+1]]
        Ca = Covariances("oas").fit_transform(sa)
        Cb = Covariances("oas").fit_transform(sb)
        ts = TangentSpace(metric="riemann").fit(Ca)
        ta.append(ts.transform(Ca)); tb.append(ts.transform(Cb))
        ma.append(sa.mean(2)); mb.append(sb.mean(2))
    return np.stack(ta, 1), np.stack(tb, 1), np.stack(ma, 1), np.stack(mb, 1)


def ev(bt, be, ytr, yte):
    sc = [StandardScaler().fit(x) for x in bt]
    A = np.hstack([s.transform(x) for s, x in zip(sc, bt)])
    B = np.hstack([s.transform(x) for s, x in zip(sc, be)])
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(A, ytr)
    return balanced_accuracy_score(yte, m.predict(B))


def main(task, seeds):
    files = [f for f in sorted(glob.glob(os.path.join(BM.DATA, "*.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    W, Win, leaks = BM.reservoir_weights(0)
    NAMES = ["cov", "cov+res", "cov+mean", "cov+meandiff", "cov+mean+diffs",
             "cov+mean+diffs+res"]
    R = {k: [] for k in NAMES}
    print("task=%s  data=%s  window %.2f-%.2f  subjects=%d\n" % (
        task, BM.DATA, BM.WMIN, BM.WMAX, len(files)))
    for path in files:
        Xb, ph, _ = BM.load_subject(path)
        X, y, n_win, chance = BM.build_task(Xb, ph, task)
        res = BM.multitimescale_reservoir(X, W, Win, leaks)
        acc = {k: [] for k in NAMES}
        for sd in range(seeds):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                ta, tb, ma, mb = parts(X[tr], X[te], n_win, BM.NPCA)
                C = (ta.reshape(len(ta), -1), tb.reshape(len(tb), -1))
                M = (ma.reshape(len(ma), -1), mb.reshape(len(mb), -1))
                CD = (np.diff(ta, axis=1).reshape(len(ta), -1),
                      np.diff(tb, axis=1).reshape(len(tb), -1))
                MD = (np.diff(ma, axis=1).reshape(len(ma), -1),
                      np.diff(mb, axis=1).reshape(len(mb), -1))
                acc["cov"].append(ev([C[0]], [C[1]], y[tr], y[te]))
                acc["cov+res"].append(ev([C[0], res[tr]], [C[1], res[te]], y[tr], y[te]))
                acc["cov+mean"].append(ev([C[0], M[0]], [C[1], M[1]], y[tr], y[te]))
                acc["cov+meandiff"].append(ev([C[0], MD[0]], [C[1], MD[1]], y[tr], y[te]))
                acc["cov+mean+diffs"].append(
                    ev([C[0], M[0], CD[0], MD[0]], [C[1], M[1], CD[1], MD[1]], y[tr], y[te]))
                acc["cov+mean+diffs+res"].append(
                    ev([C[0], M[0], CD[0], MD[0], res[tr]],
                       [C[1], M[1], CD[1], MD[1], res[te]], y[tr], y[te]))
        for k in NAMES:
            R[k].append(float(np.mean(acc[k])))
        print("  %-10s %s" % (os.path.basename(path)[:10],
              " ".join("%.3f" % R[k][-1] for k in NAMES)), flush=True)

    print("\n=== %s (chance %.3f) ===" % (task, chance))
    base = np.array(R["cov"])
    for k in NAMES:
        a = np.array(R[k]); _, p0 = ttest_1samp(a, chance); _, pb = ttest_rel(a, base)
        print("  %-20s %.3f  vs-chance p=%.7f  vs-cov %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean() - base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["place", "ph11"], default="ph11")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.task, a.seeds)
