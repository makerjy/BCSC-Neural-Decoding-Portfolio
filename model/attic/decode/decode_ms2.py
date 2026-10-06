import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
WMIN, WMAX = 0.0, 0.5

def band_power(x: np.ndarray, sf: float = 500.0) -> np.ndarray:
    from scipy.signal import welch
    n, c, T = x.shape
    f, P = welch(x, fs=sf, nperseg=min(128, T), axis=2)
    out = []
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 60), (60, 120)]:
        s = (f >= lo) & (f < hi)
        out.append(np.log(P[:, :, s].mean(2) + 1e-20))
    return np.stack(out, 2).reshape(n, -1)

volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))

def ms(F: np.ndarray, y: np.ndarray, est) -> float:
    yy = pd.Categorical(y).codes
    return float(np.mean([cross_val_score(est, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(5)]))

res = {"cat_V": [], "cat_BP": [], "pl_V": [], "pl_BP": [], "vow_V": []}
print("single-phoneme control, %.1f-%.1fs, baseline-corr | multi-seed bal-acc (no perm, fast)\n" % (WMIN, WMAX))
print("%-8s %8s %8s %8s %8s %8s" % ("subj", "catV", "catBP", "plV", "plBP", "vowV(.2)"))
for f in FILES:
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    if X.shape[1] != 61:
        print("%-8s SKIP %dch" % (sub, X.shape[1])); continue
    df = pd.DataFrame({k: d[k] for k in ["task", "tms_target", "category", "place", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    Fb = band_power(Xb)
    cat = ds.category.values
    cons = (ds.category != "vowels").values
    vow = (ds.category == "vowels").values
    cV = ms(Fv, cat, volt); cB = ms(Fb, cat, svm)
    pV = ms(Fv[cons], ds[cons].place.values, volt); pB = ms(Fb[cons], ds[cons].place.values, svm)
    vV = ms(Fv[vow], ds[vow].phoneme.values, volt)
    res["cat_V"].append(cV); res["cat_BP"].append(cB); res["pl_V"].append(pV); res["pl_BP"].append(pB); res["vow_V"].append(vV)
    print("%-8s %8.3f %8.3f %8.3f %8.3f %8.3f" % (sub, cV, cB, pV, pB, vV), flush=True)

print("\n=== GROUP (1-sample t vs chance) ===")
for k, ch, lbl in [("cat_V", .5, "category Volt"), ("cat_BP", .5, "category BP"),
                   ("pl_V", .5, "place Volt"), ("pl_BP", .5, "place BP"), ("vow_V", .2, "vowel Volt")]:
    a = np.array(res[k]); tt, p = ttest_1samp(a, ch)
    print("  %-14s mean %.3f +- %.3f  %d/%d>chance  p=%.4f %s" % (
        lbl, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > ch else ""))
