import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann.estimation import XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
LEAK = 0.3

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df[m].reset_index(drop=True)

def smooth(X3: np.ndarray, leak: float = LEAK) -> np.ndarray:
    out = np.empty_like(X3)
    s = np.zeros(X3.shape[:2])
    for ti in range(X3.shape[2]):
        s = (1 - leak) * s + leak * X3[:, :, ti]
        out[:, :, ti] = s
    return out

lda = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def pipe_xdawn(n=4):
    return make_pipeline(XdawnCovariances(nfilter=n, estimator="oas", xdawn_estimator="oas"),
                         TangentSpace(metric="riemann"),
                         StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
def pipe_cov():
    return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                         StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))

def score(est, F, y):
    return float(np.mean([cross_val_score(est, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(3)]))

NAMES = ["volt", "smooth", "cov_plain", "xdawn", "xdawn_smooth", "xdawn+smooth_concat"]
R = {k: [] for k in NAMES}
print("Riemannian done RIGHT for ERP (Xdawn prototype-augmented) vs smoothing | place-stops\n")
for f in FILES:
    Xb, ds = load(f)
    ph = ds.phoneme.values; mask = np.isin(ph, ["b", "p", "d", "t"])
    y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
    Xm = Xb[mask]
    Xs = smooth(Xm)
    R["volt"].append(score(lda, Xm[:, :, ::10].reshape(len(Xm), -1), y))
    R["smooth"].append(score(lda, Xs[:, :, ::10].reshape(len(Xs), -1), y))
    R["cov_plain"].append(score(pipe_cov(), Xm, y))
    R["xdawn"].append(score(pipe_xdawn(), Xm, y))
    R["xdawn_smooth"].append(score(pipe_xdawn(), Xs, y))
    R["xdawn+smooth_concat"].append(np.nan)
    print("  %-8s volt %.3f smooth %.3f cov %.3f xdawn %.3f xdawn_sm %.3f" % (
        os.path.basename(f).split("_")[0], R["volt"][-1], R["smooth"][-1], R["cov_plain"][-1],
        R["xdawn"][-1], R["xdawn_smooth"][-1]), flush=True)

vo = np.array(R["volt"]); sm = np.array(R["smooth"])
print("\n=== GROUP (place-stops, within-subject) ===")
for k in ["volt", "smooth", "cov_plain", "xdawn", "xdawn_smooth"]:
    a = np.array(R[k])
    _, pv = ttest_rel(a, vo); _, ps = ttest_rel(a, sm)
    print("  %-13s %.3f   vs-volt %+.3f (p=%.4f)   vs-smooth %+.3f (p=%.4f)" % (
        k, a.mean(), a.mean()-vo.mean(), pv, a.mean()-sm.mean(), ps))
