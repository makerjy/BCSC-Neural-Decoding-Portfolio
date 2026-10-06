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
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5

def load_subj(f: str) -> tuple[np.ndarray, pd.DataFrame]:
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, ds

def inv_sqrt(M: np.ndarray) -> np.ndarray:
    return np.real(fmp(M, -0.5))

def align(covs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return covs
    ref = covs.mean(0) if mode == "euclid" else mean_riemann(covs)
    W = inv_sqrt(ref)
    return np.stack([W @ c @ W for c in covs], 0)

subs = []
for f in FILES:
    Xb, ds = load_subj(f)
    C = Covariances("oas").fit_transform(Xb)
    subs.append((os.path.basename(f).split("_")[0], C, ds))

def run_target(target: str):
    print("\n=== %s | LOSO cross-subject (CIPHER place-LOSO ~0.518 chance) ===" % target)
    for mode in ["none", "euclid", "riemann"]:
        accs = []
        aligned = []
        for name, C, ds in subs:
            ph = ds.phoneme.values
            if target == "place-stops":
                mask = np.isin(ph, ["b", "p", "d", "t"]); y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
            else:
                mask = np.ones(len(ds), bool); y = np.where(ds.category.values != "vowels", 1, 0)
            aligned.append((align(C[mask], mode), y))
        for i in range(len(subs)):
            Ctr = np.concatenate([aligned[j][0] for j in range(len(subs)) if j != i], 0)
            ytr = np.concatenate([aligned[j][1] for j in range(len(subs)) if j != i], 0)
            Cte, yte = aligned[i]
            ts = TangentSpace(metric="riemann").fit(Ctr)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
            clf.fit(ts.transform(Ctr), ytr)
            accs.append(balanced_accuracy_score(yte, clf.predict(ts.transform(Cte))))
        a = np.array(accs); _, p = ttest_1samp(a, 0.5)
        print("  align=%-8s LOSO bal %.3f +- %.3f  (%d/%d>chance)  vs-chance p=%.4f %s" % (
            mode, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))

run_target("category")
run_target("place-stops")
