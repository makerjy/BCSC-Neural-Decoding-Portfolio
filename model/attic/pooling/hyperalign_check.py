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

def load(f: str):
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df.phoneme.values[m]

def template(Xb, ph, anchors):
    return np.concatenate([Xb[ph == a].mean(0)[:, ::10] for a in anchors], axis=1)

def procrustes(A, B):
    U, _, Vt = np.linalg.svd(B @ A.T); return U @ Vt

subj = [(os.path.basename(f).split("_")[0], *load(f)) for f in FILES]
VOWELS = ["a", "i", "e", "u", "o"]

def hyperalign_loso(anchors: list[str], loo_ref: bool):
    Mv = [template(Xb, ph, anchors) for _, Xb, ph in subj]
    accs = []
    for i in range(len(subj)):
        idx = [j for j in range(len(subj)) if j != i] if loo_ref else list(range(len(subj)))
        ref = Mv[idx[0]].copy()
        for _ in range(3):
            R = {j: procrustes(Mv[j], ref) for j in idx}
            ref = np.mean([R[j] @ Mv[j] for j in idx], 0)
        R = {j: procrustes(Mv[j], ref) for j in range(len(subj))}
        data = []
        for j, (_, Xb, ph) in enumerate(subj):
            mask = np.isin(ph, ["b", "p", "d", "t"])
            Xa = np.stack([R[j] @ x for x in Xb[mask]], 0)
            data.append((Xa[:, :, ::10].reshape(mask.sum(), -1), np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)))
        Ftr = np.concatenate([data[j][0] for j in idx], 0)
        ytr = np.concatenate([data[j][1] for j in idx], 0)
        clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        clf.fit(Ftr, ytr)
        accs.append(balanced_accuracy_score(data[i][1], clf.predict(data[i][0])))
    return np.array(accs)

print("=== PLACE-STOPS cross-subject LOSO | leakage checks ===\n")
for lab, anchors, loo in [
        ("stop-anchors, ref-incl-test (ORIGINAL, leaky?)", ["b", "p", "d", "t"], False),
        ("stop-anchors, LOO-ref", ["b", "p", "d", "t"], True),
        ("VOWEL-anchors, LOO-ref (CLEAN)", VOWELS, True)]:
    l = hyperalign_loso(anchors, loo)
    _, p = ttest_1samp(l, 0.5)
    print("  %-42s bal %.3f +- %.3f  (%d/%d)  p=%.4f %s" % (
        lab, l.mean(), l.std(), int((l > 0.5).sum()), len(l), p, "**SIG**" if p < 0.05 and l.mean() > 0.5 else ""))
