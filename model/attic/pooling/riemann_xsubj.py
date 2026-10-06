import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power as fmp
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    ph = df.phoneme.values[m]
    mask = np.isin(ph, ["b", "p", "d", "t"])
    return Xb[mask], np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)

def recenter(C):
    W = np.real(fmp(mean_riemann(C), -0.5))
    return np.stack([W @ c @ W for c in C], 0)

subj = []
for f in FILES:
    Xm, y = load(f)
    C = Covariances("oas").fit_transform(Xm)
    subj.append((os.path.basename(f).split("_")[0], C, y))
print("clean single-phoneme control | %d subjects | trials/subj %s\n" % (len(subj), [len(y) for _, _, y in subj]))

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))

print("=== WITHIN-subject (reference) ===")
wi = []
for _, C, y in subj:
    F = TangentSpace(metric="riemann").fit_transform(C)
    wi.append(np.mean([cross_val_score(clf(), F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                       scoring="balanced_accuracy").mean() for s in range(3)]))
wi = np.array(wi); _, p = ttest_1samp(wi, 0.5)
print("  Riemannian within: %.3f +- %.3f  (%d/%d)  p=%.4f\n" % (wi.mean(), wi.std(), int((wi > 0.5).sum()), len(wi), p))

print("=== CROSS-subject LOSO ===")
for mode in ["no-align", "recenter"]:
    data = [(recenter(C) if mode == "recenter" else C, y) for _, C, y in subj]
    accs = []
    for i in range(len(data)):
        Ctr = np.concatenate([data[j][0] for j in range(len(data)) if j != i], 0)
        ytr = np.concatenate([data[j][1] for j in range(len(data)) if j != i], 0)
        ts = TangentSpace(metric="riemann").fit(Ctr)
        c = clf(); c.fit(ts.transform(Ctr), ytr)
        accs.append(balanced_accuracy_score(data[i][1], c.predict(ts.transform(data[i][0]))))
    a = np.array(accs); _, p = ttest_1samp(a, 0.5)
    print("  %-9s LOSO: %.3f +- %.3f  (%d/%d>chance)  p=%.4f %s" % (
        mode, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))

print("\n=== PERMUTATION TEST (within-subject, recentered, 200 perms) ===")
rng = np.random.default_rng(0)
real, nulls = [], []
for _, C, y in subj[:8]:
    F = TangentSpace(metric="riemann").fit_transform(recenter(C))
    r = np.mean([cross_val_score(clf(), F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                 scoring="balanced_accuracy").mean() for s in range(2)])
    nl = [cross_val_score(clf(), F, rng.permutation(y), cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          scoring="balanced_accuracy").mean() for _ in range(25)]
    real.append(r); nulls.append(np.mean(nl))
print("  real %.3f   null %.3f   (null should be ~0.500)" % (np.mean(real), np.mean(nulls)))
