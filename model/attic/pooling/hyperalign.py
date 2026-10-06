import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
ANCHORS = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]

def load(f: str):
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df.phoneme.values[m]

def template(Xb: np.ndarray, ph: np.ndarray) -> np.ndarray:
    return np.concatenate([Xb[ph == a].mean(0)[:, ::10] if (ph == a).any()
                           else np.zeros((Xb.shape[1], len(range(0, Xb.shape[2], 10))))
                           for a in ANCHORS], axis=1)

def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(B @ A.T)
    return U @ Vt

subj = [(os.path.basename(f).split("_")[0], *load(f)) for f in FILES]
M = [template(Xb, ph) for _, Xb, ph in subj]
ref = M[0].copy()
for _ in range(3):
    Rs = [procrustes(M[i], ref) for i in range(len(M))]
    ref = np.mean([Rs[i] @ M[i] for i in range(len(M))], 0)
Rs = [procrustes(M[i], ref) for i in range(len(M))]

def run(aligned: bool):
    data = []
    for i, (name, Xb, ph) in enumerate(subj):
        mask = np.isin(ph, ["b", "p", "d", "t"])
        Xa = np.stack([Rs[i] @ x for x in Xb[mask]], 0) if aligned else Xb[mask]
        F = Xa[:, :, ::10].reshape(mask.sum(), -1)
        y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
        data.append((F, y))
    accs = []
    for i in range(len(data)):
        Ftr = np.concatenate([data[j][0] for j in range(len(data)) if j != i], 0)
        ytr = np.concatenate([data[j][1] for j in range(len(data)) if j != i], 0)
        clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        clf.fit(Ftr, ytr)
        accs.append(balanced_accuracy_score(data[i][1], clf.predict(data[i][0])))
    return np.array(accs)

print("=== PLACE-STOPS cross-subject LOSO | single-phoneme, S* ===")
for lab, al in [("no-align", False), ("HYPERALIGN", True)]:
    l = run(al); _, p = ttest_1samp(l, 0.5)
    print("  %-11s LOSO bal %.3f +- %.3f  (%d/%d>ch)  p=%.4f  %s" % (
        lab, l.mean(), l.std(), int((l > 0.5).sum()), len(l), p, "**SIG**" if p < 0.05 and l.mean() > 0.5 else ""))
