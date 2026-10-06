import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def rocket_fast(X: np.ndarray, n_kernels: int, seed: int = 0, batch: int = 250) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, C, T = X.shape
    Ls = rng.choice([7, 9, 11], n_kernels)
    dils = np.empty(n_kernels, int)
    for i, L in enumerate(Ls):
        md = max(1, int(np.log2((T - 1) / (L - 1))))
        d = int(2 ** rng.uniform(0, md))
        dils[i] = d if (L - 1) * d < T else 1
    Ws = [rng.normal(0, 1, (C, L)) for L in Ls]
    Ws = [w - w.mean() for w in Ws]
    mask = [rng.random(C) < 0.5 for _ in range(n_kernels)]
    Ws = [w * m[:, None] if m.any() else w for w, m in zip(Ws, mask)]
    biases = rng.uniform(-1, 1, n_kernels)
    feats = np.empty((n, n_kernels * 2), np.float32)
    for L in np.unique(Ls):
        for d in np.unique(dils[Ls == L]):
            idx = np.where((Ls == L) & (dils == d))[0]
            span = (L - 1) * d
            if span >= T:
                continue
            seg = np.stack([X[:, :, i*d:T - span + i*d] for i in range(L)], -1)
            for s in range(0, len(idx), batch):
                sub = idx[s:s+batch]
                Wb = np.stack([Ws[i] for i in sub], 0)
                conv = np.einsum("nctl,mcl->nmt", seg, Wb)
                b = biases[sub][None, :, None] * (np.abs(conv).mean((0, 2))[None, :, None] + 1e-9)
                z = conv - b
                feats[:, 2*sub] = (z > 0).mean(2)
                feats[:, 2*sub+1] = z.max(2)
    return feats

def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def ev(Ftr, ytr, Fte, yte):
    if Ftr.shape[1] > 2500:
        c = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-2, 4, 8)))
    else:
        c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

KERNELS = [500, 2000, 6000]
NAMES = ["winriem"] + ["RK%d" % k for k in KERNELS] + ["win+RK%d" % k for k in KERNELS]
RES = {k: [] for k in NAMES}
NULLS = []
rng_g = np.random.default_rng(0)
print("ROCKET kernel sweep (vectorised) + permutation null | 11-phoneme\n")
for f in FILES:
    Xall, ph = load(f)
    k2 = np.isin(ph, PHON)
    X = Xall[k2]; y = np.array([PHON.index(v) for v in ph[k2]])
    pcg = PCA(12).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Xp = np.stack([pcg.transform(z.T).T for z in X], 0)
    RK = {k: rocket_fast(Xp, k) for k in KERNELS}
    acc = {k: [] for k in NAMES}
    nul = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        Rtr, Rte = winriem(X[tr], X[te], 6)
        acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
        s1 = StandardScaler().fit(Rtr)
        for k in KERNELS:
            acc["RK%d" % k].append(ev(RK[k][tr], y[tr], RK[k][te], y[te]))
            s2 = StandardScaler().fit(RK[k][tr])
            a = np.hstack([s1.transform(Rtr), s2.transform(RK[k][tr])])
            b = np.hstack([s1.transform(Rte), s2.transform(RK[k][te])])
            acc["win+RK%d" % k].append(ev(a, y[tr], b, y[te]))
        s2 = StandardScaler().fit(RK[500][tr])
        a = np.hstack([s1.transform(Rtr), s2.transform(RK[500][tr])])
        b = np.hstack([s1.transform(Rte), s2.transform(RK[500][te])])
        for _ in range(5):
            nul.append(ev(a, rng_g.permutation(y[tr]), b, y[te]))
    for k in NAMES: RES[k].append(float(np.mean(acc[k])))
    NULLS.append(float(np.mean(nul)))
    print("  %-8s win %.3f | RK %.3f %.3f %.3f | win+RK %.3f %.3f %.3f | null %.3f" % (
        os.path.basename(f).split("_")[0], RES["winriem"][-1],
        *[RES["RK%d" % k][-1] for k in KERNELS], *[RES["win+RK%d" % k][-1] for k in KERNELS], NULLS[-1]), flush=True)

nullm = float(np.mean(NULLS))
print("\n=== GROUP (11-phoneme) ===")
print("  EMPIRICAL CHANCE (permuted labels): %.4f   [theoretical 1/11 = %.4f]" % (nullm, 1/11))
base = np.array(RES["winriem"])
for k in NAMES:
    a = np.array(RES[k]); _, p0 = ttest_1samp(a, nullm); _, pb = ttest_rel(a, base)
    print("  %-12s %.3f  vs-null p=%.6f  vs-winriem %+.3f (p=%.4f) %s" % (
        k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
