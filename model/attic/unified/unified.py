import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_rel, t as tdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.spatialfilters import Xdawn
import fastriem as fr
import fastlda as fl
import csd as csdmod

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
CHANCE = {"place": 0.5, "ph11": 1 / 11}
CSDS, NFS, KS = (False, True), (1, 2), (1, 2)
GMAX = 0.45
GRID = [(c, n, k) for c in CSDS for n in NFS for k in KS]


def load(path, t0, t1):
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    X = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    return (X[:, :, (t >= t0) & (t < t1)], d["phoneme1"].astype(str)[k],
            np.asarray(d["channel_positions_xyz"], float))


def augment(Xtr, ytr, Xte, nf):
    xd = Xdawn(nfilter=nf, estimator="lwf").fit(Xtr, ytr)
    V, cls = xd.filters_, np.unique(ytr)
    P = np.concatenate([V[i * nf:(i + 1) * nf] @ Xtr[ytr == c].mean(0)
                        for i, c in enumerate(cls)], 0)
    fa, fb = xd.transform(Xtr), xd.transform(Xte)
    return (np.concatenate([np.repeat(P[None], len(fa), 0), fa], 1),
            np.concatenate([np.repeat(P[None], len(fb), 0), fb], 1))


def feats(Za, Zb, K, M0):
    ed = np.linspace(0, Za.shape[2], K + 1).astype(int)
    a, b, Ms = [], [], []
    for i in range(K):
        Ca = fr.cov_oas(Za[:, :, ed[i]:ed[i + 1]])
        Cb = fr.cov_oas(Zb[:, :, ed[i]:ed[i + 1]])
        M = fr.riemann_mean(Ca, init=None if M0 is None else M0[i])
        Ms.append(M)
        a.append(fr.tangent(Ca, M))
        b.append(fr.tangent(Cb, M))
    return np.concatenate(a, 1), np.concatenate(b, 1), Ms


def clf(a, ytr, b, yte):
    sc = StandardScaler().fit(a)
    co, it, cl = fl.fit(sc.transform(a), ytr)
    return balanced_accuracy_score(yte, fl.predict(sc.transform(b), co, it, cl))


def one(job):
    path, tn, t0, t1, nseed = job
    t_start = time.time()
    X, ph, pos = load(path, t0, t1)
    L = csdmod.csd_operator(pos)
    keep = STOPS if tn == "place" else PHON
    m = np.isin(ph, keep)
    p = ph[m]
    y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
         else np.array([PHON.index(v) for v in p]))
    XR = {False: X[m], True: csdmod.apply_csd(X[m], L)}
    acc = {g: [] for g in GRID}
    dims = {}
    M0 = {g: None for g in GRID}
    for sd in range(nseed):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(XR[False], y):
            for c in CSDS:
                for nf in NFS:
                    Za, Zb = augment(XR[c][tr], y[tr], XR[c][te], nf)
                    nch, T = Za.shape[1], Za.shape[2]
                    for K in KS:
                        if nch / (T // K) > GMAX:
                            continue
                        g = (c, nf, K)
                        try:
                            a, b, M0[g] = feats(Za, Zb, K, M0[g])
                            dims[g] = a.shape[1]
                            acc[g].append(clf(a, y[tr], b, y[te]))
                        except Exception:
                            M0[g] = None
    return (os.path.basename(path)[:3], tn,
            {g: (float(np.mean(acc[g])) if acc[g] else float("nan")) for g in GRID},
            dims, time.time() - t_start)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="npz2")
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--t0", type=float, default=0.2)
    ap.add_argument("--t1", type=float, default=1.01)
    ap.add_argument("--tasks", nargs="*", default=["place", "ph11"])
    ap.add_argument("--subjects", type=int, default=0)
    A = ap.parse_args()
    FILES = sorted(glob.glob(os.path.join(A.data, "*.npz")))
    if A.subjects:
        FILES = FILES[:A.subjects]
    jobs = [(f, t, A.t0, A.t1, A.seeds) for f in FILES for t in A.tasks]
    print("UNIFIED | CSD -> xDAWN -> prototype-augmented windowed SPD -> tangent -> LDA")
    print("window %.0f-%.0f ms | %d jobs | grid %d | pool %d\n"
          % (A.t0 * 1000, A.t1 * 1000, len(jobs), len(GRID), A.pool), flush=True)
    R, D = {}, {}
    with Pool(A.pool) as pool:
        for i, (sub, tn, o, dims, el) in enumerate(pool.imap_unordered(one, jobs), 1):
            R[(sub, tn)] = o
            D.update(dims)
            va = {k: v for k, v in o.items() if not np.isnan(v)}
            print("  [%2d/%2d] %s %-5s best=%.3f cfg=%s  %.0fs"
                  % (i, len(jobs), sub, tn, max(va.values()), max(va, key=va.get), el),
                  flush=True)

    def ci(d):
        m, s, n = d.mean(), d.std(ddof=1), len(d)
        h = tdist.ppf(.975, n - 1) * s / np.sqrt(n)
        return m, m - h, m + h

    subs = sorted({s for s, _ in R})
    for tn in A.tasks:
        print("\n=== %s (chance %.3f, n=%d) ===" % (tn, CHANCE[tn], len(subs)))
        tab = {}
        for g in GRID:
            v = np.array([R[(s, tn)][g] for s in subs])
            if not np.isnan(v).any():
                tab[g] = v
        for g in sorted(tab):
            print("  csd=%-5s nf=%d K=%d  dim=%-5d  %.4f"
                  % (g[0], g[1], g[2], D.get(g, 0), tab[g].mean()))
        if not tab:
            continue
        best = max(tab, key=lambda k: tab[k].mean())
        print("  BEST %s = %.4f  (%d/%d above chance)"
              % (best, tab[best].mean(), int((tab[best] > CHANCE[tn]).sum()), len(subs)))
        for nf in NFS:
            for K in KS:
                if (True, nf, K) in tab and (False, nf, K) in tab:
                    d = tab[(True, nf, K)] - tab[(False, nf, K)]
                    m, lo, hi = ci(d)
                    print("    CSD effect nf=%d K=%d: %+.4f [%+.4f,%+.4f] p=%.4f"
                          % (nf, K, m, lo, hi,
                             ttest_rel(tab[(True, nf, K)], tab[(False, nf, K)]).pvalue))
