import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
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


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def verify_centering():
    d = np.load("npz/sub-S04_ses-02.npz", allow_pickle=True)
    X = d["X"].astype(float)[:20, :10, :100]
    C_pr = Covariances("oas").fit_transform(X)
    Xc = X - X.mean(2, keepdims=True)
    Craw = np.einsum("nct,ndt->ncd", X, X) / X.shape[2]
    Ccen = np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[2]
    r_raw = np.corrcoef(C_pr.ravel(), Craw.ravel())[0, 1]
    r_cen = np.corrcoef(C_pr.ravel(), Ccen.ravel())[0, 1]
    print("pyriemann Covariances vs manual:")
    print("  without time-centering : corr %.4f" % r_raw)
    print("  with    time-centering : corr %.4f" % r_cen)
    print("  -> pyriemann %s the within-window mean\n" % (
        "REMOVES" if r_cen > r_raw else "keeps"))


def parts(Xtr, Xte, K, npca):
    pc = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    va, vb, ma, mb = [], [], [], []
    for i in range(K):
        sa, sb = A[:, :, ed[i]:ed[i+1]], B[:, :, ed[i]:ed[i+1]]
        Ca = Covariances("oas").fit_transform(sa)
        Cb = Covariances("oas").fit_transform(sb)
        ts = TangentSpace(metric="riemann").fit(Ca)
        va.append(ts.transform(Ca)); vb.append(ts.transform(Cb))
        ma.append(sa.mean(2)); mb.append(sb.mean(2))
    return (np.stack(va, 1), np.stack(vb, 1), np.stack(ma, 1), np.stack(mb, 1))


def reservoir(X, seed=0, npca=20, n_res=200, sub=20):
    rng = np.random.default_rng(seed)
    pc = PCA(npca).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([pc.transform(z.T) for z in X], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    W = rng.normal(0, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (n_res, npca)) * 0.5
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), n_res))[None, :]
    h = np.zeros((len(U), n_res)); out = []
    for ti in range(U.shape[1]):
        h = (1 - leaks) * h + leaks * np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % sub == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(len(U), -1)


def ev(Ftr, ytr, Fte, yte):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(Ftr, ytr); return balanced_accuracy_score(yte, m.predict(Fte))


def cat(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


def flat(a):
    return a.reshape(len(a), -1)


def diff(a):
    return flat(np.diff(a, axis=1))


verify_centering()
FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
NAMES = ["cov_only", "mean_only", "cov+mean", "cov+covdiff",
         "cov+meandiff", "cov+mean+diffs"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("Restoring the discarded window means (and tangent-space differences)\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (np.isin(ph, STOPS),
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0)),
             "ph11": (np.isin(ph, PHON),
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]))}
    for tn, (m, y) in tasks.items():
        X = Xall[m]
        npca, K = CFG[tn]
        R = reservoir(X)
        acc = {k: [] for k in NAMES}
        for sd in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                va, vb, ma, mb = parts(X[tr], X[te], K, npca)
                C = (flat(va), flat(vb))
                M = (flat(ma), flat(mb))
                CD = (diff(va), diff(vb))
                MD = (diff(ma), diff(mb))
                acc["cov_only"].append(ev(C[0], y[tr], C[1], y[te]))
                acc["mean_only"].append(ev(M[0], y[tr], M[1], y[te]))
                a, b = cat([C[0], M[0]], [C[1], M[1]])
                acc["cov+mean"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([C[0], CD[0]], [C[1], CD[1]])
                acc["cov+covdiff"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([C[0], MD[0]], [C[1], MD[1]])
                acc["cov+meandiff"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([C[0], M[0], CD[0], MD[0]], [C[1], M[1], CD[1], MD[1]])
                acc["cov+mean+diffs"].append(ev(a, y[tr], b, y[te]))
        for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f)  cfg d=%d K=%d ===" % (tn, ch, *CFG[tn]))
    base = np.array(RES[tn]["cov_only"])
    for k in NAMES:
        a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-20s %.3f  vs-chance p=%.7f  vs-cov %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean() - base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
