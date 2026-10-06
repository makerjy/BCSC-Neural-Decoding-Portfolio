import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
BANK = [(2, 18), (4, 20), (7, 10), (13, 23), (18, 34)]

def dda_vec(x, tau1, tau2):
    n, c, T = x.shape; m = T - tau2 - 1
    ud = x[:, :, tau2+1:tau2+1+m] - x[:, :, tau2:tau2+m]
    a = x[:, :, tau2+1-tau1:tau2+1-tau1+m]; b = x[:, :, 1:1+m]
    D = np.stack([a, b, a*a], -1)
    DtD = np.einsum("ncmi,ncmj->ncij", D, D) + 1e-8*np.eye(3)
    Dty = np.einsum("ncmi,ncm->nci", D, ud)
    coef = np.einsum("bij,bj->bi", np.linalg.inv(DtD.reshape(-1, 3, 3)), Dty.reshape(-1, 3)).reshape(n, c, 3)
    resid = np.mean((np.einsum("ncmi,nci->ncm", D, coef) - ud)**2, 2)
    return np.concatenate([coef.reshape(n, -1), resid], 1)

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def ms(F, y, est): return np.mean([cross_val_score(est, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)])

R = {t: {"bank": [], "single": [], "volt": [], "bank+volt": []} for t in ["cat", "pl"]}
for f in FILES:
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    bank = np.concatenate([dda_vec(Xb, *g) for g in BANK], 1)
    single = dda_vec(Xb, 7, 10)
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    ph = ds.phoneme.values
    for cond, mask, yv in [("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
                            ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]
        R[cond]["bank"].append(ms(bank[mask], yy, svm))
        R[cond]["single"].append(ms(single[mask], yy, svm))
        R[cond]["volt"].append(ms(Fv[mask], yy, volt))
        from sklearn.preprocessing import StandardScaler as SS
        bv = np.concatenate([SS().fit_transform(bank[mask]), SS().fit_transform(Fv[mask])], 1)
        R[cond]["bank+volt"].append(ms(bv, yy, svm))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

print("\n=== GROUP: DDA delay-bank vs single vs voltage vs fusion ===")
for cond, lbl in [("cat", "category"), ("pl", "place-stops")]:
    bk, sg, vo, bv = (np.array(R[cond][k]) for k in ["bank", "single", "volt", "bank+volt"])
    _, p_bv = ttest_rel(bk, vo); _, p_bs = ttest_rel(bk, sg); _, p_fv = ttest_rel(bv, vo)
    print("  %-11s bank %.3f  single %.3f  volt %.3f  bank+volt %.3f" % (lbl, bk.mean(), sg.mean(), vo.mean(), bv.mean()))
    print("    bank-volt=%+.3f(p=%.3f)  bank-single=%+.3f(p=%.3f)  (bank+volt)-volt=%+.3f(p=%.3f)" % (
        bk.mean()-vo.mean(), p_bv, bk.mean()-sg.mean(), p_bs, bv.mean()-vo.mean(), p_fv))
