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
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
N_RES, D_IN, SUB, LEAK = 200, 20, 20, 0.3
TAU1, TAU2 = 4, 20

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df[m].reset_index(drop=True)

def weights(seed: int):
    rng = np.random.default_rng(seed)
    def sparse(scale):
        W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
        return W / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9) * scale
    return sparse(1.0), sparse(0.5), sparse(0.3), rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def esn(U, W, Win):
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1)

def mem_only(U, Win):
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*(U[:, ti, :] @ Win.T)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1)

def smooth_only(U):
    n, T, d = U.shape; s = np.zeros((n, d)); out = []
    for ti in range(T):
        s = (1-LEAK)*s + LEAK*U[:, ti, :]
        if ti % SUB == 0: out.append(s.copy())
    return np.stack(out, 1)

def dda_reservoir(U, A1, A2, A3, Win, alpha=0.15, quad=True):
    n, T, _ = U.shape
    hist = np.zeros((n, TAU2 + 1, N_RES)); h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h1 = hist[:, -TAU1, :]; h2 = hist[:, -TAU2, :]
        upd = h1 @ A1.T + h2 @ A2.T + U[:, ti, :] @ Win.T
        if quad:
            upd = upd + (h1 * h1) @ A3.T
        h = np.clip(h + alpha * upd, -3.0, 3.0)
        hist = np.concatenate([hist[:, 1:, :], h[:, None, :]], 1)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1)

def flat(H): return H.reshape(len(H), -1)
def measure_readout(H):
    return np.concatenate([H.mean(1), H.std(1), (H**2).mean(1), np.abs(H).mean(1)], 1)

est = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def ms(F, y):
    return float(np.mean([cross_val_score(est, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(3)]))

NAMES = ["volt", "smooth_only", "mem_only", "ESN", "DDA_res", "DDA_res_quad", "DDA_measure"]
R = {k: [] for k in NAMES}
W1, W2, W3, Win = weights(0)
Wesn = W1.copy()
print("DDA-structured reservoir vs ESN vs simple smoothing | place-stops\n")
for f in FILES:
    Xb, ds = load(f)
    ph = ds.phoneme.values; mask = np.isin(ph, ["b", "p", "d", "t"])
    y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
    p = PCA(D_IN).fit(Xb.transpose(0, 2, 1).reshape(-1, Xb.shape[1]))
    U = np.stack([p.transform(x.T) for x in Xb], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    F = {"volt": Xb[:, :, ::10].reshape(len(Xb), -1),
         "smooth_only": flat(smooth_only(U)),
         "mem_only": flat(mem_only(U, Win)),
         "ESN": flat(esn(U, Wesn, Win)),
         "DDA_res": flat(dda_reservoir(U, W1, W2, W3, Win, quad=False)),
         "DDA_res_quad": flat(dda_reservoir(U, W1, W2, W3, Win, quad=True)),
         "DDA_measure": measure_readout(dda_reservoir(U, W1, W2, W3, Win, quad=True))}
    for k in NAMES:
        R[k].append(ms(F[k][mask], y))
    print("  %s " % os.path.basename(f).split("_")[0] + " ".join("%s %.3f" % (k[:9], R[k][-1]) for k in NAMES), flush=True)

vo = np.array(R["volt"]); so = np.array(R["smooth_only"])
print("\n=== GROUP (place-stops, within-subject) ===")
for k in NAMES:
    a = np.array(R[k])
    _, pv = ttest_rel(a, vo); _, ps = ttest_rel(a, so)
    print("  %-13s %.3f  vs-volt %+.3f (p=%.3f)  vs-smooth %+.3f (p=%.3f)" % (
        k, a.mean(), a.mean()-vo.mean(), pv, a.mean()-so.mean(), ps))
