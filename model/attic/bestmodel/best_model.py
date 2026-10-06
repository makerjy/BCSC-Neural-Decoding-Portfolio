"""
BCSC EEG speech decoding - best model.

Windowed Riemannian tangent-space features, plus the first-order information the covariance
throws away.

    EEG -> PCA(d) -> split time into K windows -> per window:
             covariance -> Riemannian tangent space   (2nd-order spatial geometry)
             channel mean                             (1st-order evoked deflection)
        -> concatenate windows, optionally their consecutive differences
        -> shrinkage LDA

Why the mean matters: pyriemann's Covariances centres each window over time (verified:
correlation with a non-centred estimate 0.149, with a centred one 0.9996), so the per-window
per-channel mean voltage is discarded. At these window lengths that mean is a large part of
the evoked deflection, and it decodes on its own well above chance (place 0.574, 11-phoneme
0.152). Restoring it beats covariance alone on both tasks.

Measured on 15 subjects, single-phoneme control trials, 0-0.5 s, 3 seeds x 5-fold:

    task        chance   voltage-LDA   cov only   best
    place       0.500       0.550       0.662     0.688   (cov+mean+diffs, d=40 K=2)
    11-phoneme  0.091       0.138       0.170     0.192   (cov+mean-diff,  d=25 K=6)

The earlier reservoir add-on (11-phoneme 0.190) is no longer needed: the window means carry
the same slow-timescale information more directly and in far fewer dimensions.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

DATA = "npz"
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
WMIN, WMAX = 0.0, 0.5
BASE = (-0.2, -0.11)

CONFIG = {
    "place": dict(npca=40, n_win=2, chance=0.5,
                  use_mean=True, use_cov_diff=True, use_mean_diff=True),
    "ph11": dict(npca=25, n_win=6, chance=1 / 11,
                 use_mean=False, use_cov_diff=False, use_mean_diff=True),
}


def load_subject(path):
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]]
                         for c in ["task", "tms_target", "phoneme"]})
    keep = ((meta.task == "single-phoneme")
            & meta.tms_target.str.startswith("control")).values
    bm = (t >= BASE[0]) & (t <= BASE[1])
    wm = (t >= WMIN) & (t <= WMAX)
    Xk = X[keep]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[keep]


def window_features(Xtr, Xte, n_win, npca):
    pca = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    project = lambda Z: np.stack([pca.transform(z.T).T for z in Z], 0)
    A, B = project(Xtr), project(Xte)
    edges = np.linspace(0, A.shape[2], n_win + 1).astype(int)
    ta, tb, ma, mb = [], [], [], []
    for i in range(n_win):
        sa, sb = A[:, :, edges[i]:edges[i + 1]], B[:, :, edges[i]:edges[i + 1]]
        Ca = Covariances("oas").fit_transform(sa)
        Cb = Covariances("oas").fit_transform(sb)
        ts = TangentSpace(metric="riemann").fit(Ca)
        ta.append(ts.transform(Ca)); tb.append(ts.transform(Cb))
        ma.append(sa.mean(2)); mb.append(sb.mean(2))
    return np.stack(ta, 1), np.stack(tb, 1), np.stack(ma, 1), np.stack(mb, 1)


def assemble(tan, mean, cfg):
    blocks = [tan.reshape(len(tan), -1)]
    if cfg["use_mean"]:
        blocks.append(mean.reshape(len(mean), -1))
    if cfg["use_cov_diff"] and tan.shape[1] > 1:
        blocks.append(np.diff(tan, axis=1).reshape(len(tan), -1))
    if cfg["use_mean_diff"] and mean.shape[1] > 1:
        blocks.append(np.diff(mean, axis=1).reshape(len(mean), -1))
    return blocks


def standardise(bt, be):
    sc = [StandardScaler().fit(b) for b in bt]
    return (np.hstack([s.transform(b) for s, b in zip(sc, bt)]),
            np.hstack([s.transform(b) for s, b in zip(sc, be)]))


def classify(Ftr, ytr, Fte, yte):
    est = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    est.fit(Ftr, ytr)
    return balanced_accuracy_score(yte, est.predict(Fte))


def build_task(Xb, ph, task):
    if task == "place":
        m = np.isin(ph, STOPS)
        return Xb[m], np.where(np.isin(ph[m], ["d", "t"]), 1, 0)
    m = np.isin(ph, PHON)
    return Xb[m], np.array([PHON.index(v) for v in ph[m]])


def evaluate(task, n_seeds, data_dir):
    cfg = CONFIG[task]
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "sub-S*_ses-02.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    parts = ["cov"]
    if cfg["use_mean"]: parts.append("mean")
    if cfg["use_cov_diff"]: parts.append("cov-diff")
    if cfg["use_mean_diff"]: parts.append("mean-diff")
    print("task=%s  d=%d  K=%d  features=%s  subjects=%d\n" % (
        task, cfg["npca"], cfg["n_win"], "+".join(parts), len(files)))
    scores = []
    for path in files:
        Xb, ph = load_subject(path)
        X, y = build_task(Xb, ph, task)
        accs = []
        for seed in range(n_seeds):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                ta, tb, ma, mb = window_features(X[tr], X[te], cfg["n_win"], cfg["npca"])
                Ftr, Fte = standardise(assemble(ta, ma, cfg), assemble(tb, mb, cfg))
                accs.append(classify(Ftr, y[tr], Fte, y[te]))
        scores.append(float(np.mean(accs)))
        print("  %-8s %.3f" % (os.path.basename(path).split("_")[0], scores[-1]), flush=True)
    a = np.array(scores)
    _, p = ttest_1samp(a, cfg["chance"])
    print("\n  GROUP  %.3f +- %.3f   chance %.3f   (%d/%d above)   p=%.6f" % (
        a.mean(), a.std(), cfg["chance"], int((a > cfg["chance"]).sum()), len(a), p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["place", "ph11"], default="ph11")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    evaluate(a.task, a.seeds, a.data)
