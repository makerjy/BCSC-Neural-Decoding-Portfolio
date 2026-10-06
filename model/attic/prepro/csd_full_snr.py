import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel, norm
import mne
mne.set_log_level("ERROR")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t, ch = d["X"].astype(float), d["times"], [str(c) for c in d["ch_names"]]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0
    Xb = X[m] - X[m][:, :, b].mean(2, keepdims=True)
    return Xb, t, ch, df[m].reset_index(drop=True)

def csd(Xb, t, ch, sf=500.0):
    info = mne.create_info(ch, sf, "eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    ep = mne.EpochsArray(Xb, info, tmin=float(t[0]))
    return mne.preprocessing.compute_current_source_density(ep).get_data()

def win(Xa, t):
    w = (t >= WMIN) & (t <= WMAX)
    return Xa[:, :, w]

clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def dec(F, y, seeds=3):
    a = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(F, y):
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(F[tr], y[tr]); a.append(balanced_accuracy_score(y[te], c.predict(F[te])))
    return float(np.mean(a))

def pseudo(X3, y, k, rng):
    out_X, out_y = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n = len(idx) // k
        for g in range(n):
            out_X.append(X3[idx[g*k:(g+1)*k]].mean(0)); out_y.append(cls)
    if len(out_X) == 0:
        return None, None
    return np.stack(out_X), np.array(out_y)

def dec_pseudo(X3, y, k, seeds=5):
    rng = np.random.default_rng(0); a = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(X3, y):
            Xa, ya = pseudo(X3[tr], y[tr], k, rng)
            Xb2, yb = pseudo(X3[te], y[te], k, rng)
            if Xa is None or Xb2 is None:
                continue
            if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2 or min(np.bincount(ya)) < 2:
                continue
            Fa = Xa[:, :, ::10].reshape(len(Xa), -1); Fb = Xb2[:, :, ::10].reshape(len(Xb2), -1)
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(Fa, ya); a.append(balanced_accuracy_score(yb, c.predict(Fb)))
    return float(np.mean(a)) if a else np.nan

R = {"raw": [], "csd": []}
PSE = {k: [] for k in [1, 2, 4, 8]}
DVALS = []
print("CSD on FULL original data (500Hz, ~220 trials/subj) + SNR ceiling analysis\n")
for f in FILES:
    Xb, t, ch, ds = load(f)
    Xc = csd(Xb, t, ch)
    ph = ds.phoneme.values
    m = np.isin(ph, ["b", "p", "d", "t"])
    y = np.where(np.isin(ph[m], ["d", "t"]), 1, 0)
    Xr_w, Xc_w = win(Xb, t)[m], win(Xc, t)[m]
    R["raw"].append(dec(Xr_w[:, :, ::10].reshape(m.sum(), -1), y))
    R["csd"].append(dec(Xc_w[:, :, ::10].reshape(m.sum(), -1), y))
    for k in PSE:
        PSE[k].append(dec_pseudo(Xc_w, y, k))
    a, b = Xc_w[y == 1], Xc_w[y == 0]
    dmap = (a.mean(0) - b.mean(0)) / np.sqrt(0.5 * (a.var(0) + b.var(0)) + 1e-20)
    DVALS.append(np.abs(dmap).max())
    print("  %s raw %.3f  csd %.3f  | pseudo k=1 %.3f k=8 %.3f | max|d| %.2f" % (
        os.path.basename(f).split("_")[0], R["raw"][-1], R["csd"][-1], PSE[1][-1], PSE[8][-1], DVALS[-1]), flush=True)

print("\n=== CSD vs raw (place-stops, within-subject) ===")
for tag in ["raw", "csd"]:
    a = np.array(R[tag]); _, p = ttest_1samp(a, 0.5)
    print("  %-4s %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (tag, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))
_, pc = ttest_rel(R["csd"], R["raw"])
print("  CSD-raw = %+.3f  paired-p=%.4f  %s" % (np.mean(R["csd"]) - np.mean(R["raw"]), pc, "**CSD HELPS**" if pc < 0.05 and np.mean(R["csd"]) > np.mean(R["raw"]) else "ns"))

print("\n=== SNR ceiling: accuracy vs trials-averaged (CSD) ===")
print("  (rising curve => signal EXISTS, single-trial noise is the limit)")
for k in [1, 2, 4, 8]:
    a = np.array([v for v in PSE[k] if not np.isnan(v)])
    print("  avg %d trial(s): %.3f  (n=%d subj)" % (k, a.mean(), len(a)))
d = np.array(DVALS)
print("\n  measured max |Cohen d| (single-trial, best channel-time): %.2f +- %.2f" % (d.mean(), d.std()))
print("  => theoretical single-feature Bayes accuracy: %.3f" % norm.cdf(d.mean() / 2))
