import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61][:6]
WMIN, WMAX = 0.0, 0.5
TAU1, TAU2 = 4, 20

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df[m].reset_index(drop=True)

def fit_dda(x, tau1=TAU1, tau2=TAU2):
    n, c, T = x.shape; m = T - tau2 - 1
    ud = x[:, :, tau2+1:tau2+1+m] - x[:, :, tau2:tau2+m]
    a = x[:, :, tau2+1-tau1:tau2+1-tau1+m]; b = x[:, :, 1:1+m]
    D = np.stack([a, b, a*a], -1)
    DtD = np.einsum("ncmi,ncmj->ncij", D, D) + 1e-10*np.eye(3)
    Dty = np.einsum("ncmi,ncm->nci", D, ud)
    coef = np.einsum("bij,bj->bi", np.linalg.inv(DtD.reshape(-1, 3, 3)), Dty.reshape(-1, 3)).reshape(n, c, 3)
    pred = np.einsum("ncmi,nci->ncm", D, coef)
    ss_res = ((pred - ud)**2).sum(2); ss_tot = ((ud - ud.mean(2, keepdims=True))**2).sum(2) + 1e-20
    r2 = 1 - ss_res/ss_tot
    lin_mag = np.abs(coef[:, :, 0])[:, :, None]*np.abs(a) + np.abs(coef[:, :, 1])[:, :, None]*np.abs(b)
    nl_mag = np.abs(coef[:, :, 2])[:, :, None]*(a*a)
    nl_frac = (nl_mag.sum(2) / (lin_mag.sum(2) + nl_mag.sum(2) + 1e-20))
    return coef, r2, nl_frac

def max_root(a1: float, a2: float, tau1: int = TAU1, tau2: int = TAU2) -> float:
    p = np.zeros(tau2 + 2)
    p[0] = 1.0
    p[1] = -1.0
    p[tau2 + 1 - tau1] -= a1
    p[tau2 + 1] -= a2
    r = np.roots(p)
    return float(np.abs(r).max())

print("DDA fitted-model diagnostics  (tau1=%d, tau2=%d, %d subjects)\n" % (TAU1, TAU2, len(FILES)))
ALL = {"a1": [], "a2": [], "a3": [], "r2": [], "nl": [], "rho": [], "cls": []}
for f in FILES:
    Xb, ds = load(f)
    coef, r2, nl = fit_dda(Xb)
    y = np.where(ds.category.values != "vowels", 1, 0)
    rng = np.random.default_rng(0)
    tri = rng.choice(len(Xb), min(40, len(Xb)), replace=False)
    chi = rng.choice(Xb.shape[1], 12, replace=False)
    for i in tri:
        for c in chi:
            ALL["a1"].append(coef[i, c, 0]); ALL["a2"].append(coef[i, c, 1]); ALL["a3"].append(coef[i, c, 2])
            ALL["r2"].append(r2[i, c]); ALL["nl"].append(nl[i, c])
            ALL["rho"].append(max_root(coef[i, c, 0], coef[i, c, 1]))
            ALL["cls"].append(y[i])

A = {k: np.array(v) for k, v in ALL.items()}
print("=== coefficient distributions (per trial x channel, n=%d) ===" % len(A["a1"]))
for k in ["a1", "a2", "a3"]:
    print("  %-3s  mean %+.5f  sd %.5f  |median| %.5f" % (k, A[k].mean(), A[k].std(), np.median(np.abs(A[k]))))
print("\n=== how well does the DDE describe the signal? ===")
print("  R^2 of fit:  mean %.3f  median %.3f  (1 = perfect)" % (A["r2"].mean(), np.median(A["r2"])))
print("\n=== is the NONLINEAR (chaos-capable) term actually used? ===")
print("  nonlinear share of dynamics: mean %.4f  median %.4f  (0 = purely linear)" % (A["nl"].mean(), np.median(A["nl"])))
print("  fraction of fits where nonlinear share > 10%%: %.3f" % (A["nl"] > 0.10).mean())
print("\n=== DYNAMICAL REGIME (max |root| of linearised map) ===")
print("  rho = max|z|:  mean %.4f  median %.4f" % (A["rho"].mean(), np.median(A["rho"])))
print("  fraction EXPANDING (rho > 1, chaos-capable): %.4f" % (A["rho"] > 1.0).mean())
print("  fraction CONTRACTING (rho < 1, stable):      %.4f" % (A["rho"] < 1.0).mean())
print("  rho percentiles: 5%% %.4f  50%% %.4f  95%% %.4f" % tuple(np.percentile(A["rho"], [5, 50, 95])))
print("\n=== do these differ between classes (vowel vs consonant)? ===")
for k in ["a1", "a2", "a3", "rho", "nl", "r2"]:
    t_, p_ = ttest_ind(A[k][A["cls"] == 1], A[k][A["cls"] == 0])
    d_ = (A[k][A["cls"] == 1].mean() - A[k][A["cls"] == 0].mean()) / (A[k].std() + 1e-20)
    print("  %-4s  cohen_d %+.4f  p=%.2e %s" % (k, d_, p_, "*" if p_ < 0.05 else ""))
