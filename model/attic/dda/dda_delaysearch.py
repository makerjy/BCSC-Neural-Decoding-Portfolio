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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
TAU1S = [2, 4, 6, 9, 13, 18]
DELTAS = [2, 5, 10, 16]
GRID = [(t1, t1 + dl) for t1 in TAU1S for dl in DELTAS]

def dda_vec(x: np.ndarray, tau1: int, tau2: int) -> np.ndarray:
    n, c, T = x.shape; m = T - tau2 - 1
    u = x
    ud = u[:, :, tau2 + 1:tau2 + 1 + m] - u[:, :, tau2:tau2 + m]
    a = u[:, :, tau2 + 1 - tau1:tau2 + 1 - tau1 + m]
    b = u[:, :, 1:1 + m]
    D = np.stack([a, b, a * a], axis=-1)
    DtD = np.einsum("ncmi,ncmj->ncij", D, D) + 1e-8 * np.eye(3)
    Dty = np.einsum("ncmi,ncm->nci", D, ud)
    coef = np.einsum("bij,bj->bi", np.linalg.inv(DtD.reshape(-1, 3, 3)), Dty.reshape(-1, 3)).reshape(n, c, 3)
    resid = np.mean((np.einsum("ncmi,nci->ncm", D, coef) - ud) ** 2, axis=2)
    return np.concatenate([coef.reshape(n, -1), resid], axis=1)

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def cvscore(F, y, seed):
    return cross_val_score(svm, F, y, cv=StratifiedKFold(4, shuffle=True, random_state=seed), scoring="balanced_accuracy").mean()

def nested(feats: dict, y: np.ndarray):
    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    accs, chosen = [], []
    for tr, te in outer.split(np.zeros(len(y)), y):
        best, bg = -1, GRID[0]
        for g in GRID:
            s = cvscore(feats[g][tr], y[tr], 1)
            if s > best: best, bg = s, g
        chosen.append(bg)
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
        clf.fit(feats[bg][tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(feats[bg][te])))
    return np.mean(accs), chosen

R = {t: {"fixed": [], "nested": [], "volt": []} for t in ["cat", "pl"]}
CHOSEN = []
t0 = time.time()
print("DDA delay-search (nested CV) vs fixed-DDA(7,10) vs voltage | %d grid pts, %d subj\n" % (len(GRID), len(FILES)))
for idx, f in enumerate(FILES, 1):
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    feats = {g: dda_vec(Xb, *g) for g in GRID}
    fixed = dda_vec(Xb, 7, 10)
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    ph = ds.phoneme.values
    for cond, mask, yv in [
        ("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
        ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]
        R[cond]["fixed"].append(np.mean([cross_val_score(svm, fixed[mask], yy, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)]))
        acc, ch = nested({g: feats[g][mask] for g in GRID}, yy)
        R[cond]["nested"].append(acc)
        if cond == "pl": CHOSEN += ch
        R[cond]["volt"].append(np.mean([cross_val_score(volt, Fv[mask], yy, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)]))
    print("  %s pl: fixed %.3f nested %.3f volt %.3f  [%d/%d %.1fm]" % (
        os.path.basename(f).split("_")[0], R["pl"]["fixed"][-1], R["pl"]["nested"][-1], R["pl"]["volt"][-1], idx, len(FILES), (time.time()-t0)/60), flush=True)

print("\n=== GROUP ===")
for cond, lbl in [("cat", "category"), ("pl", "place-stops")]:
    fx, ns, vo = (np.array(R[cond][k]) for k in ["fixed", "nested", "volt"])
    _, p_nf = ttest_rel(ns, fx); _, p_nv = ttest_rel(ns, vo); _, p_nc = ttest_1samp(ns, 0.5)
    print("  %-11s fixed %.3f  nested-DDA %.3f (p_ch=%.3f)  volt %.3f | nested-fixed=%+.3f(p=%.3f) nested-volt=%+.3f(p=%.3f)" % (
        lbl, fx.mean(), ns.mean(), p_nc, vo.mean(), ns.mean()-fx.mean(), p_nf, ns.mean()-vo.mean(), p_nv))
from collections import Counter
print("\n  place delay-pairs chosen (nested):", Counter(CHOSEN).most_common(6))
