import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5; BMIN, BMAX = -0.2, -0.11

def load(f):
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    bm = (t >= BMIN) & (t <= BMAX); wm = (t >= WMIN) & (t <= WMAX)
    def feat(mask): Xm = X[mask]; return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]
    ph = df.phoneme.values
    ctrl = ((df.task == "single-phoneme") & df.tms_target.str.startswith("control")).values
    tms = (df.task == "phonemes") & df.tms_target.isin(["lip", "tongue"]).values
    tc = (df.task == "phonemes") & df.tms_target.isin(["control_lip", "control_tongue"]).values
    return feat(ctrl), ph[ctrl], np.abs(feat(tms).mean(0).mean(1) - feat(tc).mean(0).mean(1))

subj = []
for f in FILES:
    Xc, phc, loc = load(f); sm = np.isin(phc, ["b", "p", "d", "t"])
    if sm.sum() >= 20 and min(np.bincount(np.isin(phc[sm], ["d", "t"]).astype(int))) >= 8:
        subj.append((Xc[sm], np.where(np.isin(phc[sm], ["d", "t"]), 1, 0), loc))

def acc(Xb, y, ch):
    F = Xb[:, ch, ::10].reshape(len(Xb), -1)
    clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    return np.mean([cross_val_score(clf, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)])

for K in [15, 20, 30]:
    tms_acc, rnd_acc = [], []
    for Xb, y, loc in subj:
        top = np.argsort(loc)[-K:]
        tms_acc.append(acc(Xb, y, top))
        rng = np.random.default_rng(0)
        rnd_acc.append(np.mean([acc(Xb, y, rng.choice(61, K, replace=False)) for _ in range(10)]))
    ta, ra = np.array(tms_acc), np.array(rnd_acc)
    _, p = ttest_rel(ta, ra)
    print("K=%2d  TMS-select %.3f  vs  random(avg10) %.3f  diff=%+.3f  paired-p=%.4f  %s  [%d/%d TMS>rand]" % (
        K, ta.mean(), ra.mean(), ta.mean()-ra.mean(), p, "**TMS-SELECTION MATTERS**" if p < 0.05 and ta.mean() > ra.mean() else "ns",
        int((ta > ra).sum()), len(subj)), flush=True)
