import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
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
D_PCA = 20

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def featurize(Xtr, Xte):
    p = PCA(D_PCA).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    def proj(Z): return np.stack([p.transform(z.T).T for z in Z], 0)
    Ctr = Covariances("oas").fit_transform(proj(Xtr))
    Cte = Covariances("oas").fit_transform(proj(Xte))
    ts = TangentSpace(metric="riemann").fit(Ctr)
    return ts.transform(Ctr), ts.transform(Cte)

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))

def transfer(Xa, ya, Xb_, yb):
    Ftr, Fte = featurize(Xa, Xb_)
    c = clf(); c.fit(Ftr, ya)
    return balanced_accuracy_score(yb, c.predict(Fte))

def within_cv(X, y, seeds=3):
    a = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(X, y):
            Ftr, Fte = featurize(X[tr], X[te])
            c = clf(); c.fit(Ftr, y[tr])
            a.append(balanced_accuracy_score(y[te], c.predict(Fte)))
    return float(np.mean(a))

R = {"ref_all": [], "voiced_only": [], "voiceless_only": [], "v2vl": [], "vl2v": []}
print("ACOUSTIC-ABSTRACTION TEST: does place decoding transfer across voicing?\n")
print("  train on b vs d (voiced)  ->  test on p vs t (voiceless), and reverse\n")
for f in FILES:
    Xm, ph = load(f)
    stops = np.isin(ph, ["b", "p", "d", "t"])
    Xs, ps = Xm[stops], ph[stops]
    y_all = np.where(np.isin(ps, ["d", "t"]), 1, 0)
    vd = np.isin(ps, ["b", "d"]); vl = np.isin(ps, ["p", "t"])
    if vd.sum() < 16 or vl.sum() < 16:
        print("  %s skip" % os.path.basename(f).split("_")[0]); continue
    Xv, yv = Xs[vd], np.where(ps[vd] == "d", 1, 0)
    Xu, yu = Xs[vl], np.where(ps[vl] == "t", 1, 0)
    R["ref_all"].append(within_cv(Xs, y_all))
    R["voiced_only"].append(within_cv(Xv, yv))
    R["voiceless_only"].append(within_cv(Xu, yu))
    R["v2vl"].append(transfer(Xv, yv, Xu, yu))
    R["vl2v"].append(transfer(Xu, yu, Xv, yv))
    print("  %-8s ref %.3f | voiced %.3f voiceless %.3f | v->vl %.3f  vl->v %.3f" % (
        os.path.basename(f).split("_")[0], R["ref_all"][-1], R["voiced_only"][-1],
        R["voiceless_only"][-1], R["v2vl"][-1], R["vl2v"][-1]), flush=True)

print("\n=== GROUP ===")
for k, lbl in [("ref_all", "place, all stops (reference)"), ("voiced_only", "place within voiced (b vs d)"),
               ("voiceless_only", "place within voiceless (p vs t)"),
               ("v2vl", "TRANSFER voiced -> voiceless"), ("vl2v", "TRANSFER voiceless -> voiced")]:
    a = np.array(R[k]); _, p = ttest_1samp(a, 0.5)
    print("  %-32s %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (
        lbl, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))
tr = np.concatenate([R["v2vl"], R["vl2v"]])
_, p = ttest_1samp(tr, 0.5)
print("\n  BOTH TRANSFER DIRECTIONS POOLED: %.3f  p=%.4f  %s" % (
    tr.mean(), p, "ABSTRACT PLACE CODE" if p < 0.05 and tr.mean() > 0.5 else "no abstraction evidence"))
