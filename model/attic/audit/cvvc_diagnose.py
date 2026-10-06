import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans
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

def epoch(X, t, mask, t0, t1):
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def cv_decode(Xw, y, seeds=3):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 8:
        return np.nan
    accs = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(Xw, y):
            pc = PCA(min(NPCA, len(tr)-2)).fit(Xw[tr].transpose(0, 2, 1).reshape(-1, Xw.shape[1]))
            prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
            A, B = prj(Xw[tr]), prj(Xw[te])
            T = A.shape[2]; ed = np.linspace(0, T, K_WIN+1).astype(int)
            fa, fb = [], []
            for i in range(K_WIN):
                Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
                Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
                ts = TangentSpace(metric="riemann").fit(Ca)
                fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(np.hstack(fa), y[tr])
            accs.append(balanced_accuracy_score(y[te], c.predict(np.hstack(fb))))
    return float(np.mean(accs))

pc_tms, syl_all, cl_a, cl_b, cl_best, splits = [], [], [], [], [], []
print("A) positive control: decode TMS condition from syllable epochs")
print("B) label-free clustering of syllable trials (early ERP) -> place decoding within cluster\n")
for f in FILES:
    X, t, meta = load(f)
    ph = meta.phoneme.values; task = meta.task.values; tms = meta.tms_target.values
    m_tms = (task == "phonemes") & np.isin(tms, ["lip", "tongue"]) & np.isin(ph, STOPS)
    if m_tms.sum() >= 40:
        Xt = epoch(X, t, m_tms, 0.0, 0.5)
        yt = (tms[m_tms] == "lip").astype(int)
        pc_tms.append(cv_decode(Xt, yt, seeds=2))
    ctrl = np.array([s.startswith("control") for s in tms])
    m_syl = (task == "phonemes") & np.isin(ph, STOPS) & ctrl
    if m_syl.sum() < 40:
        print("  %s skip" % os.path.basename(f).split("_")[0]); continue
    Xs = epoch(X, t, m_syl, 0.0, 0.5)
    y_pl = np.where(np.isin(ph[m_syl], ["d", "t"]), 1, 0)
    syl_all.append(cv_decode(Xs, y_pl))
    early = epoch(X, t, m_syl, 0.0, 0.18)
    E = PCA(6).fit_transform(early.reshape(len(early), -1))
    lab = KMeans(2, n_init=10, random_state=0).fit_predict(E)
    frac = lab.mean()
    splits.append(min(frac, 1-frac))
    a0 = cv_decode(Xs[lab == 0], y_pl[lab == 0], seeds=3)
    a1 = cv_decode(Xs[lab == 1], y_pl[lab == 1], seeds=3)
    cl_a.append(a0); cl_b.append(a1)
    cl_best.append(np.nanmax([a0, a1]))
    print("  %-8s TMS-ctrl %.3f | syl-all %.3f | cluster split %d/%d -> place %.3f / %.3f" % (
        os.path.basename(f).split("_")[0], pc_tms[-1] if pc_tms else np.nan, syl_all[-1],
        int((lab == 0).sum()), int((lab == 1).sum()), a0, a1), flush=True)

def rep(name, arr, ch):
    a = np.array([v for v in arr if not np.isnan(v)])
    _, p = ttest_1samp(a, ch)
    print("  %-34s %.3f +- %.3f  (%d/%d>ch)  p=%.5f %s" % (
        name, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))

print("\n=== RESULTS ===")
rep("A) TMS lip-vs-tongue (positive ctrl)", pc_tms, 0.5)
rep("B) syllable place, all trials", syl_all, 0.5)
rep("B) syllable place, cluster 0", cl_a, 0.5)
rep("B) syllable place, cluster 1", cl_b, 0.5)
rep("B) syllable place, better cluster", cl_best, 0.5)
print("  mean minority-cluster fraction: %.2f (0.5 = balanced split, consistent with CV/VC)" % np.mean(splits))
