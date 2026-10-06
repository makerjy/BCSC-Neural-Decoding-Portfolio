import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, chi2_contingency
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
STOPS = ["b", "p", "d", "t"]
K_WIN, NPCA = 2, 20

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def feats(Xtr, Xte):
    pc = PCA(min(NPCA, len(Xtr)-2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = prj(Xtr), prj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K_WIN+1).astype(int)
    fa, fb = [], []
    for i in range(K_WIN):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def fit_eval(Xtr, ytr, Xte, yte):
    a, b = feats(Xtr, Xte)
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(a, ytr); return balanced_accuracy_score(yte, c.predict(b))

def random_cv(Xw, y, seeds=3):
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(Xw, y):
            acc.append(fit_eval(Xw[tr], y[tr], Xw[te], y[te]))
    return float(np.mean(acc))

def blocked_cv(Xw, y, n_blocks=4):
    n = len(y); acc = []
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    for i in range(n_blocks):
        te = np.arange(edges[i], edges[i+1]); tr = np.setdiff1d(np.arange(n), te)
        if len(np.unique(y[te])) < 2 or min(np.bincount(y[tr])) < 5: continue
        acc.append(fit_eval(Xw[tr], y[tr], Xw[te], y[te]))
    return float(np.mean(acc)) if acc else np.nan

R = {k: [] for k in ["rand", "blocked", "prestim", "order"]}
chi_p = []
print("CONFOUND AUDIT on isolated-phoneme place decoding\n")
for f in FILES:
    X, t, meta = load(f)
    ph = meta.phoneme.values; task = meta.task.values; tms = meta.tms_target.values
    ctrl = np.array([s.startswith("control") for s in tms])
    m = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    if m.sum() < 40: continue
    idx = np.where(m)[0]
    y = np.where(np.isin(ph[m], ["d", "t"]), 1, 0)
    ct = pd.crosstab(ph[m], tms[m])
    chi_p.append(chi2_contingency(ct)[1] if ct.shape[1] > 1 else np.nan)
    bm = (t >= -0.2) & (t <= -0.11); post = (t >= 0.0) & (t <= 0.5)
    Xall = X[m]
    Xpost = (Xall - Xall[:, :, bm].mean(2, keepdims=True))[:, :, post]
    R["rand"].append(random_cv(Xpost, y))
    R["blocked"].append(blocked_cv(Xpost, y))
    pre = (t >= -0.2) & (t <= -0.05)
    Xpre = Xall[:, :, pre] - Xall[:, :, pre].mean(2, keepdims=True)
    R["prestim"].append(random_cv(Xpre, y, seeds=2))
    half = (np.arange(m.sum()) > m.sum()//2).astype(int)
    R["order"].append(random_cv(Xpost, half, seeds=2))
    print("  %-8s rand %.3f | blocked %.3f | PRE-stim %.3f | early-vs-late %.3f | chi2 p=%.3f" % (
        os.path.basename(f).split("_")[0], R["rand"][-1], R["blocked"][-1],
        R["prestim"][-1], R["order"][-1], chi_p[-1]), flush=True)

print("\n=== AUDIT RESULTS ===")
for k, lbl, ch in [("rand", "place, random CV (standard)", 0.5),
                   ("blocked", "place, CHRONOLOGICAL CV", 0.5),
                   ("prestim", "place from PRE-STIMULUS window", 0.5),
                   ("order", "trial order (early vs late) decodable", 0.5)]:
    a = np.array([v for v in R[k] if not np.isnan(v)])
    _, p = ttest_1samp(a, ch)
    print("  %-38s %.3f +- %.3f  (%d/%d)  p=%.5f %s" % (
        lbl, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))
cp = np.array([v for v in chi_p if not np.isnan(v)])
print("  phoneme x TMS-target independence: %d/%d subjects p<0.05 (want few)" % (int((cp < 0.05).sum()), len(cp)))
