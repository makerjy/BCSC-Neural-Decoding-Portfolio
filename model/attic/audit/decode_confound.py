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
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
WMIN, WMAX = 0.0, 0.5
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def ms(F: np.ndarray, y: np.ndarray) -> float:
    yy = pd.Categorical(y).codes
    return float(np.mean([cross_val_score(volt, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(5)]))

res = {"cat": [], "place_all": [], "place_stops": [], "manner": []}
print("single-phoneme control, %.1f-%.1fs, Voltage+shrinkLDA | CONFOUND-CONTROLLED\n" % (WMIN, WMAX))
print("%-8s %10s %10s %12s %10s" % ("subj", "cat(vC).5", "placeAll.5", "placeSTOPS.5", "manner.5"))
for f in FILES:
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    if X.shape[1] != 61:
        continue
    df = pd.DataFrame({k: d[k] for k in ["task", "tms_target", "category", "place", "phoneme", "manner"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    F = Xb[:, :, ::10].reshape(len(Xb), -1)
    ph = ds.phoneme.values
    is_cons = ds.category.values != "vowels"
    y_cat = np.where(is_cons, "cons", "vowel")
    stops = np.isin(ph, ["b", "p", "d", "t"])
    y_place_stops = np.where(np.isin(ph, ["d", "t"]), "alv", "bil")
    cons_mask = is_cons
    y_place_all = ds.place.values
    y_manner = ds.manner.values
    cat = ms(F, y_cat)
    pa = ms(F[cons_mask], y_place_all[cons_mask])
    ps = ms(F[stops], y_place_stops[stops])
    mn_mask = np.isin(y_manner, ["stop", "fricative"])
    mn = ms(F[mn_mask], y_manner[mn_mask]) if mn_mask.sum() > 20 else np.nan
    res["cat"].append(cat); res["place_all"].append(pa); res["place_stops"].append(ps); res["manner"].append(mn)
    print("%-8s %10.3f %10.3f %12.3f %10.3f" % (sub, cat, pa, ps, mn), flush=True)

print("\n=== GROUP (1-sample t vs chance 0.5) ===")
for k, lbl in [("cat", "category vowel-vs-cons"), ("place_all", "place ALL (manner-confound)"),
               ("place_stops", "place STOPS-only (clean)"), ("manner", "manner fric-vs-stop")]:
    a = np.array(res[k]); a = a[~np.isnan(a)]
    tt, p = ttest_1samp(a, 0.5)
    print("  %-30s mean %.3f +- %.3f  %d/%d>chance  p=%.4f %s" % (
        lbl, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))
