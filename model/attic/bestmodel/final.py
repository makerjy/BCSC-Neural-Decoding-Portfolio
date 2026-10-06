import warnings
warnings.filterwarnings("ignore")
import argparse
import csv
import glob
import os
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_1samp, t as tdist
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import fastriem as fr
import fastlda as fl

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
CHANCE = {"place": 0.5, "ph11": 1 / 11}
CFG = {"place": (40, 2), "ph11": (25, 6)}
T0, T1 = 0.0, 1.01


def load(path):
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    X = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    m = (t >= T0) & (t < T1)
    return X[:, :, m], d["phoneme1"].astype(str)[k]


def run(X, y, d, K, seeds, perm_seed=None):
    if perm_seed is not None:
        y = np.random.default_rng(perm_seed).permutation(y)
    acc, M0 = [], None
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            pc = PCA(min(d, len(tr) - 2)).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
            pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
            A, B = pr(X[tr]), pr(X[te])
            ed = np.linspace(0, A.shape[2], K + 1).astype(int)
            fa, fb, Ms = [], [], []
            for i in range(K):
                Ca = fr.cov_oas(A[:, :, ed[i]:ed[i + 1]])
                Cb = fr.cov_oas(B[:, :, ed[i]:ed[i + 1]])
                M = fr.riemann_mean(Ca, init=None if M0 is None else M0[i])
                Ms.append(M)
                fa.append(fr.tangent(Ca, M))
                fb.append(fr.tangent(Cb, M))
            M0 = Ms
            a, b = np.concatenate(fa, 1), np.concatenate(fb, 1)
            sc = StandardScaler().fit(a)
            co, it, cl = fl.fit(sc.transform(a), y[tr])
            acc.append(balanced_accuracy_score(y[te], fl.predict(sc.transform(b), co, it, cl)))
    return float(np.mean(acc)), a.shape[1]


def one(job):
    path, tn, nseed, nperm = job
    t0 = time.time()
    X, ph = load(path)
    d, K = CFG[tn]
    keep = STOPS if tn == "place" else PHON
    m = np.isin(ph, keep)
    p = ph[m]
    y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
         else np.array([PHON.index(v) for v in p]))
    real, dim = run(X[m], y, d, K, range(nseed))
    perm = [run(X[m], y, d, K, [0], perm_seed=900 + i)[0] for i in range(nperm)]
    return (os.path.basename(path)[:3], tn, real, float(np.mean(perm)),
            float(np.std(perm, ddof=1)), dim, int(m.sum()), time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--perms", type=int, default=10)
    A = ap.parse_args()
    F = sorted(glob.glob("npz2/*.npz"))
    jobs = [(f, tk, A.seeds, A.perms) for f in F for tk in ("place", "ph11")]
    print("FINAL | single pre-specified model, no config selection")
    print("  PCA(61->d) -> K windows over 0-1000ms -> OAS cov -> Riemannian tangent -> dual LDA")
    print("  ph11 d=25 K=6 | place d=40 K=2 | %d CV seeds | %d permutations\n"
          % (A.seeds, A.perms), flush=True)
    rows = []
    with Pool(A.pool) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append(r)
            print("  [%2d/%2d] %s %-5s real=%.4f  perm=%.4f+-%.4f  dim=%d n=%d  %.0fs"
                  % (i, len(jobs), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]), flush=True)
    with open("final_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "task", "real", "perm_mean", "perm_sd", "dim", "n_trials", "sec"])
        w.writerows(rows)
    print("\nsaved final_results.csv")
    for tn in ("place", "ph11"):
        v = np.array([r[2] for r in rows if r[1] == tn])
        pm = np.array([r[3] for r in rows if r[1] == tn])
        n = len(v)
        h = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
        print("\n=== %s (chance %.4f, n=%d) ===" % (tn, CHANCE[tn], n))
        print("  real  %.4f  95%%CI [%.4f, %.4f]   p_vs_chance=%.3e"
              % (v.mean(), v.mean() - h, v.mean() + h, ttest_1samp(v, CHANCE[tn]).pvalue))
        print("  perm  %.4f  (null; deviation from chance %+.4f)" % (pm.mean(),
                                                                     pm.mean() - CHANCE[tn]))
        print("  above chance: %d/%d   min %.3f (%s)  max %.3f (%s)"
              % (int((v > CHANCE[tn]).sum()), n, v.min(),
                 [r[0] for r in rows if r[1] == tn][int(v.argmin())], v.max(),
                 [r[0] for r in rows if r[1] == tn][int(v.argmax())]))
