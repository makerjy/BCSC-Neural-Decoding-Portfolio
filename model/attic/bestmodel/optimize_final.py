import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
CONFIGS = [(1, 20, 1.0), (2, 20, 1.0), (3, 20, 1.0), (4, 20, 1.0), (4, 15, 1.0),
           (6, 15, 1.0), (6, 20, 1.0), (8, 12, 1.0), (4, 20, 0.5), (6, 15, 0.5)]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def winriem(Xtr, Xte, K, npca, stride_frac):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; L = int(T / K); S = max(1, int(L * stride_frac))
    starts = list(range(0, T - L + 1, S))
    fa, fb = [], []
    for s0 in starts:
        Ca = Covariances("oas").fit_transform(A[:, :, s0:s0+L])
        Cb = Covariances("oas").fit_transform(B[:, :, s0:s0+L])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))

RES = {t: {c: [] for c in CONFIGS} for t in ["place", "ph11"]}
print("PHASE 1: window hyperparameter sweep (K, npca, stride)\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0)),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]))}
    for tname, (X, y) in tasks.items():
        for cfg in CONFIGS:
            a = []
            for s in range(2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                    Ftr, Fte = winriem(X[tr], X[te], *cfg)
                    c = lr(); c.fit(Ftr, y[tr]); a.append(balanced_accuracy_score(y[te], c.predict(Fte)))
            RES[tname][cfg].append(float(np.mean(a)))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

best = {}
for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    rows = sorted(CONFIGS, key=lambda c: -np.mean(RES[tname][c]))
    for cfg in rows:
        a = np.array(RES[tname][cfg]); _, p = ttest_1samp(a, ch)
        print("  K=%d npca=%2d stride=%.1f  %.3f  p=%.6f" % (*cfg, a.mean(), p))
    best[tname] = rows[0]
    print("  BEST: K=%d npca=%d stride=%.1f -> %.3f" % (*rows[0], np.mean(RES[tname][rows[0]])))

print("\n\nPHASE 2: classifier comparison on best config\n")
CLFS = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)),
        "lda_shrink": lambda: make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")),
        "linSVM_C0.01": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.01, class_weight="balanced", max_iter=5000)),
        "linSVM_C0.1": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)),
        "linSVM_C1": lambda: make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)),
        "rbfSVM_C1_g.001": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma=1e-3, class_weight="balanced")),
        "rbfSVM_C10_g.0001": lambda: make_pipeline(StandardScaler(), SVC(C=10.0, gamma=1e-4, class_weight="balanced"))}
R2 = {t: {k: [] for k in CLFS} for t in ["place", "ph11"]}
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0)),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]))}
    for tname, (X, y) in tasks.items():
        acc = {k: [] for k in CLFS}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Ftr, Fte = winriem(X[tr], X[te], *best[tname])
                for k, mk in CLFS.items():
                    c = mk(); c.fit(Ftr, y[tr]); acc[k].append(balanced_accuracy_score(y[te], c.predict(Fte)))
        for k in CLFS: R2[tname][k].append(float(np.mean(acc[k])))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s classifiers (chance %.3f, config K=%d npca=%d) ===" % (tname, ch, best[tname][0], best[tname][1]))
    base = np.array(R2[tname]["logreg"])
    for k in CLFS:
        a = np.array(R2[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-18s %.3f  vs-chance p=%.6f  vs-logreg %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
