"""
BCSC EEG speech decoding — best model.

Core: windowed Riemannian tangent-space features.
  EEG -> PCA -> split time into K windows -> covariance per window
      -> Riemannian tangent space -> concatenate -> linear classifier

Optional auxiliary representations (concatenated, each standardised):
  - multi-timescale reservoir (heterogeneous leak rates)
  - ROCKET (dilated random convolutions + PPV/max pooling)

Best measured (15 subjects, single-phoneme control trials, 0-0.5 s):
  place (b,p vs d,t)   chance 0.500 : 0.637 core / 0.650 tuned
  11-phoneme           chance 0.091 : 0.177 core / 0.197 core+reservoir

Usage:
  python best_model.py                 # 11-phoneme, core+reservoir
  python best_model.py --task place
  python best_model.py --aux core rocket
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

DATA = "npz2"
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
WMIN, WMAX = 0.0, 1.01
BASE = (-1.0, 0.02)
NPCA = 20
K_PLACE, K_PH11 = 2, 6
N_RES, SUB_RES, D_RES_IN = 200, 20, 20


def load_subject(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme1"]})
    keep = ((meta.task == "single_phoneme") & (meta.tms_target == "control")).values
    bm = (t >= BASE[0]) & (t <= BASE[1])
    wm = (t >= WMIN) & (t <= WMAX)
    Xk = X[keep]
    Xb = (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm]
    return Xb, meta.phoneme1.values[keep], t[wm]


def window_parts(Xtr, Xte, n_win, n_pca=NPCA):
    pca = PCA(min(n_pca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    project = lambda Z: np.stack([pca.transform(z.T).T for z in Z], 0)
    A, B = project(Xtr), project(Xte)
    edges = np.linspace(0, A.shape[2], n_win + 1).astype(int)
    ta, tb, ma, mb = [], [], [], []
    for i in range(n_win):
        sa, sb = A[:, :, edges[i]:edges[i+1]], B[:, :, edges[i]:edges[i+1]]
        Ca = Covariances("oas").fit_transform(sa)
        Cb = Covariances("oas").fit_transform(sb)
        ts = TangentSpace(metric="riemann").fit(Ca)
        ta.append(ts.transform(Ca)); tb.append(ts.transform(Cb))
        ma.append(sa.mean(2)); mb.append(sb.mean(2))
    return np.stack(ta, 1), np.stack(tb, 1), np.stack(ma, 1), np.stack(mb, 1)


def windowed_riemann(Xtr: np.ndarray, Xte: np.ndarray, n_win: int,
                     n_pca: int = NPCA) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(min(n_pca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    project = lambda Z: np.stack([pca.transform(z.T).T for z in Z], 0)
    A, B = project(Xtr), project(Xte)
    edges = np.linspace(0, A.shape[2], n_win + 1).astype(int)
    fa, fb = [], []
    for i in range(n_win):
        Ca = Covariances("oas").fit_transform(A[:, :, edges[i]:edges[i + 1]])
        Cb = Covariances("oas").fit_transform(B[:, :, edges[i]:edges[i + 1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca))
        fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)


def reservoir_weights(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (N_RES, D_RES_IN)) * 0.5
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), N_RES))[None, :]
    return W, Win, leaks


def multitimescale_reservoir(X: np.ndarray, W: np.ndarray, Win: np.ndarray,
                             leaks: np.ndarray) -> np.ndarray:
    pca = PCA(D_RES_IN).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([pca.transform(z.T) for z in X], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    h = np.zeros((len(U), N_RES))
    states = []
    for ti in range(U.shape[1]):
        h = (1 - leaks) * h + leaks * np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % SUB_RES == 0:
            states.append(h.copy())
    return np.stack(states, 1).reshape(len(U), -1)


def rocket(X: np.ndarray, n_kernels: int = 1500, seed: int = 0,
           batch: int = 250) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, C, T = X.shape
    lengths = rng.choice([7, 9, 11], n_kernels)
    dil = np.array([int(2 ** rng.uniform(0, max(1, int(np.log2((T - 1) / (L - 1))))))
                    for L in lengths])
    dil = np.where((lengths - 1) * dil < T, dil, 1)
    kernels = []
    for L in lengths:
        w = rng.normal(0, 1, (C, L))
        w -= w.mean()
        m = rng.random(C) < 0.5
        kernels.append(w * m[:, None] if m.any() else w)
    biases = rng.uniform(-1, 1, n_kernels)
    out = np.empty((n, n_kernels * 2), np.float32)
    for L in np.unique(lengths):
        for d in np.unique(dil[lengths == L]):
            idx = np.where((lengths == L) & (dil == d))[0]
            span = (L - 1) * d
            if span >= T:
                continue
            seg = np.stack([X[:, :, i * d:T - span + i * d] for i in range(L)], -1)
            for s in range(0, len(idx), batch):
                sub = idx[s:s + batch]
                Wb = np.stack([kernels[i] for i in sub], 0)
                conv = np.einsum("nctl,mcl->nmt", seg, Wb)
                b = biases[sub][None, :, None] * (np.abs(conv).mean((0, 2))[None, :, None] + 1e-9)
                z = conv - b
                out[:, 2 * sub] = (z > 0).mean(2)
                out[:, 2 * sub + 1] = z.max(2)
    return out


def stack_blocks(train_blocks: list[np.ndarray],
                 test_blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scalers = [StandardScaler().fit(b) for b in train_blocks]
    return (np.hstack([s.transform(b) for s, b in zip(scalers, train_blocks)]),
            np.hstack([s.transform(b) for s, b in zip(scalers, test_blocks)]))


def classify(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray, yte: np.ndarray) -> float:
    est = (make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-2, 4, 8)))
           if Ftr.shape[1] > 2500
           else make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")))
    est.fit(Ftr, ytr)
    return balanced_accuracy_score(yte, est.predict(Fte))


def build_task(Xb: np.ndarray, ph: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray, int, float]:
    if task == "place":
        m = np.isin(ph, STOPS)
        return Xb[m], np.where(np.isin(ph[m], ["d", "t"]), 1, 0), K_PLACE, 0.5
    m = np.isin(ph, PHON)
    return Xb[m], np.array([PHON.index(v) for v in ph[m]]), K_PH11, 1 / 11



VARIANTS = ["cov", "cov+mean", "cov+meandiff", "cov+mean+diffs", "cov+res", "cov+mean+res"]


def flat(a):
    return a.reshape(len(a), -1)


def run_subject(args):
    path, task, n_seeds = args
    Xb, ph, _ = load_subject(path)
    X, y, n_win, _ = build_task(Xb, ph, task)
    W, Win, leaks = reservoir_weights(0)
    res = multitimescale_reservoir(X, W, Win, leaks)
    acc = {v: [] for v in VARIANTS}
    for seed in range(n_seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            ta, tb, ma, mb = window_parts(X[tr], X[te], n_win)
            C = (flat(ta), flat(tb)); M = (flat(ma), flat(mb))
            MD = (flat(np.diff(ma, 1)), flat(np.diff(mb, 1)))
            CD = (flat(np.diff(ta, 1)), flat(np.diff(tb, 1)))
            R = (res[tr], res[te])
            combos = {
                "cov": ([C[0]], [C[1]]),
                "cov+mean": ([C[0], M[0]], [C[1], M[1]]),
                "cov+meandiff": ([C[0], MD[0]], [C[1], MD[1]]),
                "cov+mean+diffs": ([C[0], M[0], CD[0], MD[0]], [C[1], M[1], CD[1], MD[1]]),
                "cov+res": ([C[0], R[0]], [C[1], R[1]]),
                "cov+mean+res": ([C[0], M[0], R[0]], [C[1], M[1], R[1]]),
            }
            for name, (bt, be) in combos.items():
                Ftr, Fte = stack_blocks(bt, be) if len(bt) > 1 else (bt[0], be[0])
                acc[name].append(classify(Ftr, y[tr], Fte, y[te]))
    return os.path.basename(path).split("_")[0], {v: float(np.mean(acc[v])) for v in VARIANTS}


def main(task, n_seeds, workers, data_dir):
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "*.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    chance = 0.5 if task == "place" else 1 / 11
    print("task=%s  subjects=%d  seeds=%d  workers=%d" % (task, len(files), n_seeds, workers))
    print("  " + "  ".join(VARIANTS) + chr(10), flush=True)
    out = {v: [] for v in VARIANTS}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, sc in ex.map(run_subject, [(f, task, n_seeds) for f in files]):
            for v in VARIANTS: out[v].append(sc[v])
            print("  %-6s %s" % (name, "  ".join("%.3f" % sc[v] for v in VARIANTS)), flush=True)
    print(chr(10) + "=== %s (chance %.3f) ===" % (task, chance))
    base = np.array(out["cov"])
    for v in VARIANTS:
        a = np.array(out[v])
        _, p0 = ttest_1samp(a, chance); _, pb = ttest_rel(a, base)
        print("  %-16s %.3f +- %.3f  vs-chance p=%.7f  vs-cov %+.3f (p=%.4f) %s" % (
            v, a.mean(), a.std(), p0, a.mean()-base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["place", "ph11"], default="ph11")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    main(a.task, a.seeds, a.workers, a.data)
