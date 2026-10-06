import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power as fmp
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

FILES = sorted(glob.glob("data/npz/sub-*.npz"))
WMIN, WMAX = 0.0, 0.5
BMIN, BMAX = -0.2, -0.11

print("channel counts:")
keep = []
for f in FILES:
    d = np.load(f, allow_pickle=True)
    nch = d["X"].shape[1]
    tasks = set(str(x) for x in d["task"])
    print("  %-22s %d ch  %d trials  tasks=%s" % (os.path.basename(f), nch, d["X"].shape[0], tasks))
    if nch == 61:
        keep.append(f)
print("\n61-ch subjects usable: %d / %d\n" % (len(keep), len(FILES)))

def build(f, target):
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "place", "phoneme"]})
    use = np.isin(df.task.values, ["single-phoneme", "phonemes"])
    ph = df.phoneme.values
    if target == "place-stops":
        mask = use & np.isin(ph, ["b", "p", "d", "t"])
        y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
    else:
        mask = use & np.isin(df.place.values, ["alveolar", "bilabial", "vowels"])
        y = np.where(df.place.values[mask] != "vowels", 1, 0)
    Xm = X[mask]
    bm = (t >= BMIN) & (t <= BMAX); wm = (t >= WMIN) & (t <= WMAX)
    Xb = (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]
    return Covariances("oas").fit_transform(Xb), y

def inv_sqrt(M): return np.real(fmp(M, -0.5))
def recenter(C):
    W = inv_sqrt(mean_riemann(C)); return np.stack([W @ c @ W for c in C], 0)

for target in ["category", "place-stops"]:
    data = []
    for f in keep:
        C, y = build(f, target)
        if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 10:
            data.append((os.path.basename(f).split("_")[0], recenter(C), y))
    ns = [len(y) for _, _, y in data]
    print("=== %s | %d subjects, trials/subj mean %d (total %d) ===" % (target, len(data), int(np.mean(ns)), sum(ns)))
    within = []
    for name, C, y in data:
        ts = TangentSpace(metric="riemann")
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        F = ts.fit_transform(C)
        within.append(np.mean([cross_val_score(clf, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)]))
    loso = []
    for i in range(len(data)):
        Ctr = np.concatenate([data[j][1] for j in range(len(data)) if j != i], 0)
        ytr = np.concatenate([data[j][2] for j in range(len(data)) if j != i], 0)
        ts = TangentSpace(metric="riemann").fit(Ctr)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        clf.fit(ts.transform(Ctr), ytr)
        loso.append(balanced_accuracy_score(data[i][2], clf.predict(ts.transform(data[i][1]))))
    w, l = np.array(within), np.array(loso)
    _, pw = ttest_1samp(w, 0.5); _, pl = ttest_1samp(l, 0.5)
    print("  WITHIN-subject  bal %.3f +- %.3f  (%d/%d>ch)  p=%.4f" % (w.mean(), w.std(), int((w > 0.5).sum()), len(w), pw))
    print("  CROSS-subj LOSO bal %.3f +- %.3f  (%d/%d>ch)  p=%.4f  %s\n" % (l.mean(), l.std(), int((l > 0.5).sum()), len(l), pl, "**SIG**" if pl < 0.05 and l.mean() > 0.5 else ""))
