import warnings
warnings.filterwarnings("ignore")
import glob
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
D_PCA, D_RFF = 8, 400

def dda(x, tau1=7, tau2=10):
    n, c, T = x.shape; mm = T-tau2-1; o = np.empty((n, c, 4), np.float32)
    for i in range(n):
        for j in range(c):
            u = x[i, j]; ud = u[tau2+1:tau2+1+mm]-u[tau2:tau2+mm]
            a = u[tau2+1-tau1:tau2+1-tau1+mm]; bb = u[1:1+mm]; D = np.stack([a, bb, a*a], 1)
            cf, *_ = np.linalg.lstsq(D, ud, rcond=None); o[i, j] = [cf[0], cf[1], cf[2], float(np.mean((D@cf-ud)**2))]
    return o.reshape(n, -1)

def measure_embed(Xb: np.ndarray, delays: list[int]) -> np.ndarray:
    n, c, T = Xb.shape
    flat = Xb.transpose(0, 2, 1).reshape(-1, c)
    proj = PCA(D_PCA).fit_transform(flat).reshape(n, T, D_PCA)
    mu, sd = proj.reshape(-1, D_PCA).mean(0), proj.reshape(-1, D_PCA).std(0) + 1e-8
    proj = (proj - mu) / sd
    maxd = max(delays); dim = D_PCA * len(delays)
    allpts = []
    ptslist = []
    for i in range(n):
        pts = np.concatenate([proj[i, maxd-dl:T-dl] for dl in delays], axis=1)
        ptslist.append(pts); allpts.append(pts[::3])
    rbf = RBFSampler(gamma=1.0/dim, n_components=D_RFF, random_state=0)
    rbf.fit(np.vstack(allpts))
    return np.stack([rbf.transform(p).mean(0) for p in ptslist], 0)

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def ms(F, y, est):
    return float(np.mean([cross_val_score(est, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(5)]))

R = {t: {"volt": [], "dda": [], "mdyn": [], "mstat": []} for t in ["cat", "pl"]}
t0 = time.time()
print("Dynamics-in-kernel (measure-embedding) vs DDA-features vs voltage | %d subj\n" % len(FILES))
print("%-7s | place-stops: volt  ddaF  mDYN  mStat" % "subj")
for idx, f in enumerate(FILES, 1):
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    Fd = dda(Xb)
    Mdyn = measure_embed(Xb, [0, 4, 8])
    Mstat = measure_embed(Xb, [0])
    ph = ds.phoneme.values
    for cond, mask, yv in [
        ("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
        ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]
        R[cond]["volt"].append(ms(Fv[mask], yy, volt))
        R[cond]["dda"].append(ms(Fd[mask], yy, svm))
        R[cond]["mdyn"].append(ms(Mdyn[mask], yy, svm))
        R[cond]["mstat"].append(ms(Mstat[mask], yy, svm))
    r = R["pl"]
    print("%-7s | %.3f %.3f %.3f %.3f  [%d/%d %.1fm]" % (
        sub, r["volt"][-1], r["dda"][-1], r["mdyn"][-1], r["mstat"][-1], idx, len(FILES), (time.time()-t0)/60), flush=True)

print("\n=== GROUP (vs chance 0.5, and measure-dyn vs others paired-t) ===")
for cond, lbl in [("cat", "category"), ("pl", "place-stops")]:
    r = {k: np.array(v) for k, v in R[cond].items()}
    for k in ["volt", "dda", "mdyn", "mstat"]:
        _, p = ttest_1samp(r[k], 0.5)
        print("  %-11s %-6s %.3f (p_chance=%.3f)" % (lbl, k, r[k].mean(), p))
    _, p1 = ttest_rel(r["mdyn"], r["volt"]); _, p2 = ttest_rel(r["mdyn"], r["dda"]); _, p3 = ttest_rel(r["mdyn"], r["mstat"])
    print("    mDYN-volt=%+.3f(p=%.3f)  mDYN-ddaF=%+.3f(p=%.3f)  mDYN-mStat=%+.3f(p=%.3f) <-dynamics-in-kernel effect" % (
        r["mdyn"].mean()-r["volt"].mean(), p1, r["mdyn"].mean()-r["dda"].mean(), p2, r["mdyn"].mean()-r["mstat"].mean(), p3))
