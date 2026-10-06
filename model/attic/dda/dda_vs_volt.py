import warnings
warnings.filterwarnings("ignore")
import glob
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
WMIN, WMAX = 0.0, 0.5

def dda(x: np.ndarray, tau1: int = 7, tau2: int = 10) -> np.ndarray:
    n, c, T = x.shape
    mm = T - tau2 - 1
    o = np.empty((n, c, 4), np.float32)
    for i in range(n):
        for j in range(c):
            u = x[i, j]
            ud = u[tau2 + 1:tau2 + 1 + mm] - u[tau2:tau2 + mm]
            a = u[tau2 + 1 - tau1:tau2 + 1 - tau1 + mm]
            bb = u[1:1 + mm]
            D = np.stack([a, bb, a * a], 1)
            cf, *_ = np.linalg.lstsq(D, ud, rcond=None)
            o[i, j] = [cf[0], cf[1], cf[2], float(np.mean((D @ cf - ud) ** 2))]
    return o.reshape(n, -1)

volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))

def ms(F: np.ndarray, y: np.ndarray, est) -> float:
    yy = pd.Categorical(y).codes
    return float(np.mean([cross_val_score(est, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(5)]))

res = {"cat_V": [], "cat_D": [], "pl_V": [], "pl_D": []}
t0 = time.time()
subs = [f for f in FILES if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
print("DDA vs Voltage-LDA on signal-bearing data | %d subjects, %.1f-%.1fs\n" % (len(subs), WMIN, WMAX))
print("%-8s %8s %8s | %10s %10s" % ("subj", "catV", "catDDA", "plStopsV", "plStopsDDA"))
for k, f in enumerate(subs, 1):
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0
    w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    Fd = dda(Xb)
    ph = ds.phoneme.values
    y_cat = np.where(ds.category.values != "vowels", "cons", "vowel")
    stops = np.isin(ph, ["b", "p", "d", "t"])
    y_ps = np.where(np.isin(ph, ["d", "t"]), "alv", "bil")
    cV = ms(Fv, y_cat, volt); cD = ms(Fd, y_cat, svm)
    pV = ms(Fv[stops], y_ps[stops], volt); pD = ms(Fd[stops], y_ps[stops], svm)
    res["cat_V"].append(cV); res["cat_D"].append(cD); res["pl_V"].append(pV); res["pl_D"].append(pD)
    eta = (time.time() - t0) / k * (len(subs) - k)
    print("%-8s %8.3f %8.3f | %10.3f %10.3f   [%d/%d ETA %.1fm]" % (
        sub, cV, cD, pV, pD, k, len(subs), eta / 60), flush=True)

print("\n=== GROUP (mean +- sd, vs chance 0.5, and DDA-vs-Voltage paired t) ===")
for lbl, kv, kd in [("category", "cat_V", "cat_D"), ("place-stops", "pl_V", "pl_D")]:
    av, ad = np.array(res[kv]), np.array(res[kd])
    _, pv = ttest_1samp(av, 0.5); _, pd_ = ttest_1samp(ad, 0.5)
    _, pcmp = ttest_rel(ad, av)
    print("  %-12s Volt %.3f (p=%.4f)  DDA %.3f (p=%.4f)  | DDA-Volt=%+.3f paired-p=%.4f %s" % (
        lbl, av.mean(), pv, ad.mean(), pd_, ad.mean() - av.mean(), pcmp,
        "DDA>Volt" if ad.mean() > av.mean() and pcmp < 0.05 else ("Volt>DDA" if av.mean() > ad.mean() and pcmp < 0.05 else "tie")))
