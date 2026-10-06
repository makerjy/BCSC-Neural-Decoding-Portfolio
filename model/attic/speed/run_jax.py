import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import ttest_1samp
from sklearn.model_selection import StratifiedKFold

from jax_pipeline import eval_fold, reservoir_run

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
N_RES, D_IN, SUB = 200, 20, 20


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11)
    wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def reservoir_weights(seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (N_RES, D_IN)) * 0.5
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), N_RES))[None, :]
    return jnp.asarray(W), jnp.asarray(Win), jnp.asarray(leaks)


def reservoir_features(X: np.ndarray, W, Win, leaks) -> jnp.ndarray:
    flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    flat = flat - flat.mean(0)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    U = np.einsum("kc,nct->ntk", Vt[:D_IN], X)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    return reservoir_run(jnp.asarray(U), W, Win, leaks, SUB)


def build(X: np.ndarray, ph: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    if task == "place":
        m = np.isin(ph, STOPS)
        return X[m], np.where(np.isin(ph[m], ["d", "t"]), 1, 0), 2, 2, 0.5
    m = np.isin(ph, PHON)
    return X[m], np.array([PHON.index(v) for v in ph[m]]), 6, 11, 1 / 11


def main(task: str, use_res: bool, seeds: int, n_pca: int) -> None:
    files = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    W, Win, leaks = reservoir_weights(0)
    scores = []
    t0 = time.time()
    print("JAX pipeline | task=%s reservoir=%s seeds=%d npca=%d\n" % (task, use_res, seeds, n_pca))
    for path in files:
        Xb, ph = load(path)
        X, y, n_win, n_class, chance = build(Xb, ph, task)
        Xj = jnp.asarray(X)
        R = reservoir_features(X, W, Win, leaks) if use_res else None
        accs = []
        for s in range(seeds):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                a = eval_fold(Xj[tr], jnp.asarray(y[tr]), Xj[te], jnp.asarray(y[te]),
                              n_win, n_pca, n_class,
                              R[tr] if R is not None else None,
                              R[te] if R is not None else None)
                accs.append(float(a))
        scores.append(float(np.mean(accs)))
        print("  %-8s %.3f   [%.1fs]" % (os.path.basename(path).split("_")[0],
                                          scores[-1], time.time() - t0), flush=True)
    a = np.array(scores)
    _, p = ttest_1samp(a, chance)
    print("\n  GROUP %.3f +- %.3f  chance %.3f  (%d/%d)  p=%.6f" % (
        a.mean(), a.std(), chance, int((a > chance).sum()), len(a), p))
    print("  total %.1fs (%.1fs/subject)" % (time.time() - t0, (time.time() - t0) / len(files)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["place", "ph11"], default="ph11")
    ap.add_argument("--reservoir", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--npca", type=int, default=20)
    args = ap.parse_args()
    main(args.task, args.reservoir, args.seeds, args.npca)
