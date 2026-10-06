import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
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
BMIN, BMAX = -0.2, -0.11
STOPS = ["b", "p", "d", "t"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    bm = (t >= BMIN) & (t <= BMAX); wm = (t >= WMIN) & (t <= WMAX)
    Xb = (X - X[:, :, bm].mean(2, keepdims=True))[:, :, wm]
    return Xb, df

def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def ev(Xtr, ytr, Xte, yte, K):
    a, b = winriem(Xtr, Xte, K)
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(a, ytr); return balanced_accuracy_score(yte, c.predict(b))

def run(X, y, K, seeds=3):
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            acc.append(ev(X[tr], y[tr], X[te], y[te], K))
    return float(np.mean(acc))

SETS = ["single_only", "single+phonemes_ctrl", "all_tasks_ctrl", "ALL_TASKS_ALL_TMS"]
R = {t: {s: [] for s in SETS} for t in ["place2", "stop4"]}
NTR = {s: [] for s in SETS}
print("MORE PER-SUBJECT DATA: pooling tasks for stop-consonant decoding\n")
for f in FILES:
    Xb, df = load(f)
    ph = df.phoneme.values; task = df.task.values; tms = df.tms_target.values
    is_stop = np.isin(ph, STOPS)
    ctrl = np.array([str(v).startswith("control") for v in tms])
    masks = {
        "single_only": is_stop & (task == "single-phoneme"),
        "single+phonemes_ctrl": is_stop & np.isin(task, ["single-phoneme", "phonemes"]) & ctrl,
        "all_tasks_ctrl": is_stop & ctrl,
        "ALL_TASKS_ALL_TMS": is_stop,
    }
    line = []
    for s in SETS:
        m = masks[s]
        Xs = Xb[m]; p = ph[m]
        y2 = np.where(np.isin(p, ["d", "t"]), 1, 0)
        y4 = np.array([STOPS.index(v) for v in p])
        NTR[s].append(int(m.sum()))
        if m.sum() < 40 or min(np.bincount(y4)) < 5:
            R["place2"][s].append(np.nan); R["stop4"][s].append(np.nan); line.append("  n/a  "); continue
        R["place2"][s].append(run(Xs, y2, 2))
        R["stop4"][s].append(run(Xs, y4, 4))
        line.append("%3d:%.3f/%.3f" % (m.sum(), R["place2"][s][-1], R["stop4"][s][-1]))
    print("  %-8s %s" % (os.path.basename(f).split("_")[0], " | ".join(line)), flush=True)

for tname, ch in [("place2", 0.5), ("stop4", 0.25)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(R[tname]["single_only"])
    for s in SETS:
        a = np.array(R[tname][s]); ok = ~np.isnan(a) & ~np.isnan(base)
        _, p0 = ttest_1samp(a[~np.isnan(a)], ch)
        _, pb = ttest_rel(a[ok], base[ok])
        print("  %-22s n=%4.0f  %.3f  vs-chance p=%.6f  vs-single %+.3f (p=%.4f) %s" % (
            s, np.mean(NTR[s]), np.nanmean(a), p0, np.nanmean(a)-np.nanmean(base), pb,
            "**BEATS**" if pb < 0.05 and np.nanmean(a) > np.nanmean(base) else ""))
