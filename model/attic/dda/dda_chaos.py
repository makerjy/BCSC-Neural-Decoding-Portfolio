import warnings
warnings.filterwarnings("ignore")
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
BANK = [(2, 18), (4, 20), (7, 10), (18, 34)]

def dda_dynamics(x, tau1, tau2):
    n, c, T = x.shape; m = T - tau2 - 1
    xl = x[:, :, tau2+1-tau1:tau2+1-tau1+m]
    ud = x[:, :, tau2+1:tau2+1+m] - x[:, :, tau2:tau2+m]
    a = xl; b = x[:, :, 1:1+m]
    D = np.stack([a, b, a*a], -1)
    DtD = np.einsum("ncmi,ncmj->ncij", D, D) + 1e-8*np.eye(3)
    Dty = np.einsum("ncmi,ncm->nci", D, ud)
    coef = np.einsum("bij,bj->bi", np.linalg.inv(DtD.reshape(-1, 3, 3)), Dty.reshape(-1, 3)).reshape(n, c, 3)
    a1, a2, a3 = coef[:, :, 0], coef[:, :, 1], coef[:, :, 2]
    local_exp = np.mean(np.abs(a1[:, :, None] + 2*a3[:, :, None]*xl), axis=2)
    fp_stab = np.abs(a1 + a2)
    nonlin = np.abs(a3) * xl.std(2)
    return np.stack([local_exp, fp_stab, nonlin], -1).reshape(n, -1)

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def ms(F, y, est): return np.mean([cross_val_score(est, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)])

R = {t: {"chaos": [], "volt": [], "chaos+volt": []} for t in ["cat", "pl"]}
for f in FILES:
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    chaos = np.concatenate([dda_dynamics(Xb, *g) for g in BANK], 1)
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    ph = ds.phoneme.values
    for cond, mask, yv in [("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
                            ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]
        R[cond]["chaos"].append(ms(chaos[mask], yy, svm))
        R[cond]["volt"].append(ms(Fv[mask], yy, volt))
        cv = np.concatenate([StandardScaler().fit_transform(chaos[mask]), StandardScaler().fit_transform(Fv[mask])], 1)
        R[cond]["chaos+volt"].append(ms(cv, yy, svm))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

print("\n=== GROUP: DDA chaos/dynamical-behavior features (truest test of 'chaos power') ===")
for cond, lbl in [("cat", "category"), ("pl", "place-stops")]:
    ch, vo, cv = (np.array(R[cond][k]) for k in ["chaos", "volt", "chaos+volt"])
    _, p1 = ttest_rel(ch, vo); _, p2 = ttest_rel(cv, vo)
    print("  %-11s chaos-feat %.3f  volt %.3f  chaos+volt %.3f | chaos-volt=%+.3f(p=%.3f) (chaos+volt)-volt=%+.3f(p=%.3f)" % (
        lbl, ch.mean(), vo.mean(), cv.mean(), ch.mean()-vo.mean(), p1, cv.mean()-vo.mean(), p2))
