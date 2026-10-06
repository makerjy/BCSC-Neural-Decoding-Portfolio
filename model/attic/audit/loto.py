import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_rel, t as tdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
CFG = {"place": (40, 2), "ph11": (25, 6)}
CHANCE = {"place": 0.5, "ph11": 1 / 11}
NREP = 10


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    X = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    return X, d["phoneme1"].astype(str)[k], d["stimulus"].astype(str)[k]


def cov_feat(Xtr, Xte, K, npca):
    pc = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    va, vb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i + 1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i + 1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        va.append(ts.transform(Ca))
        vb.append(ts.transform(Cb))
    return np.concatenate(va, 1), np.concatenate(vb, 1)


def ev(Xtr, ytr, Xte, yte, K, npca):
    if len(np.unique(ytr)) < 2 or len(yte) < 2:
        return np.nan
    a, b = cov_feat(Xtr, Xte, K, npca)
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(a, ytr)
    return balanced_accuracy_score(yte, m.predict(b))


def one(f):
    X, ph, st = load(f)
    sub = os.path.basename(f)[:3]
    out = {}
    for tn, keep in [("place", STOPS), ("ph11", PHON)]:
        npca, K = CFG[tn]
        m = np.isin(ph, keep)
        Xs, p, s = X[m], ph[m], st[m]
        y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
             else np.array([PHON.index(v) for v in p]))
        rng = np.random.default_rng(0)
        cv, loto, mr, tid = [], [], [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
            cv.append(ev(Xs[tr], y[tr], Xs[te], y[te], K, npca))
        toks = {c: sorted(set(s[p == c])) for c in np.unique(p)}
        for _ in range(NREP):
            hold = {c: rng.choice(toks[c]) for c in toks}
            te = np.array([s[i] == hold[p[i]] for i in range(len(p))])
            if te.sum() < 4 or (~te).sum() < 20:
                continue
            loto.append(ev(Xs[~te], y[~te], Xs[te], y[te], K, npca))
            idx = np.zeros(len(p), bool)
            for c in np.unique(p):
                j = np.where(p == c)[0]
                idx[rng.choice(j, int((te & (p == c)).sum()), replace=False)] = True
            mr.append(ev(Xs[~idx], y[~idx], Xs[idx], y[idx], K, npca))
        if tn == "ph11":
            for c in np.unique(p):
                j = np.where(p == c)[0]
                yt = np.array([toks[c].index(v) for v in s[j]])
                if len(np.unique(yt)) < 2 or min(np.bincount(yt)) < 4:
                    continue
                v = []
                for tr, te2 in StratifiedKFold(4, shuffle=True, random_state=0).split(j, yt):
                    v.append(ev(Xs[j][tr], yt[tr], Xs[j][te2], yt[te2], K, npca))
                tid.append((np.nanmean(v), 1.0 / len(toks[c])))
        out[tn] = (np.nanmean(cv), np.nanmean(loto), np.nanmean(mr),
                   np.nanmean([a for a, _ in tid]) if tid else np.nan,
                   np.nanmean([b for _, b in tid]) if tid else np.nan)
    return sub, out


if __name__ == "__main__":
    FILES = sorted(glob.glob("npz2/*.npz"))
    print("LOTO acoustic control | %d subjects | window 0-1.0s\n" % len(FILES), flush=True)
    with Pool(16) as pool:
        res = pool.map(one, FILES)
    for sub, o in res:
        print("  %s  place cv=%.3f loto=%.3f mr=%.3f | ph11 cv=%.3f loto=%.3f mr=%.3f tokID=%.3f"
              % (sub, o["place"][0], o["place"][1], o["place"][2],
                 o["ph11"][0], o["ph11"][1], o["ph11"][2], o["ph11"][3]), flush=True)

    def ci(d):
        d = d[~np.isnan(d)]
        m, s, n = d.mean(), d.std(ddof=1), len(d)
        h = tdist.ppf(.975, n - 1) * s / np.sqrt(n)
        return m, m - h, m + h

    for tn in ["place", "ph11"]:
        A = np.array([o[tn] for _, o in res], float)
        print("\n=== %s (chance %.3f) ===" % (tn, CHANCE[tn]))
        for i, nm in enumerate(["cv_standard", "LOTO", "matched_random"]):
            print("  %-16s %.3f" % (nm, np.nanmean(A[:, i])))
        m, lo, hi = ci(A[:, 1] - A[:, 2])
        print("  LOTO - matched_random  %+.3f  95%%CI [%+.3f,%+.3f]  p=%.4f"
              % (m, lo, hi, ttest_rel(A[:, 1], A[:, 2], nan_policy="omit").pvalue))
        if tn == "ph11":
            print("  token-ID within phoneme  %.3f  (chance %.3f)"
                  % (np.nanmean(A[:, 3]), np.nanmean(A[:, 4])))
