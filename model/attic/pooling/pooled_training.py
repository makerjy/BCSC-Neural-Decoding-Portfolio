import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power as fmp
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
K_WIN, NPCA = 6, 20

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def rocket_fast(X, n_kernels, seed=0, batch=250):
    rng = np.random.default_rng(seed)
    n, C, T = X.shape
    Ls = rng.choice([7, 9, 11], n_kernels)
    dils = np.array([int(2 ** rng.uniform(0, max(1, int(np.log2((T-1)/(L-1)))))) for L in Ls])
    dils = np.where((Ls-1)*dils < T, dils, 1)
    Ws = []
    for L in Ls:
        w = rng.normal(0, 1, (C, L)); w -= w.mean()
        m_ = rng.random(C) < 0.5
        Ws.append(w * m_[:, None] if m_.any() else w)
    biases = rng.uniform(-1, 1, n_kernels)
    feats = np.empty((n, n_kernels*2), np.float32)
    for L in np.unique(Ls):
        for d in np.unique(dils[Ls == L]):
            idx = np.where((Ls == L) & (dils == d))[0]
            span = (L-1)*d
            if span >= T: continue
            seg = np.stack([X[:, :, i*d:T-span+i*d] for i in range(L)], -1)
            for s in range(0, len(idx), batch):
                sub = idx[s:s+batch]
                Wb = np.stack([Ws[i] for i in sub], 0)
                conv = np.einsum("nctl,mcl->nmt", seg, Wb)
                b_ = biases[sub][None, :, None]*(np.abs(conv).mean((0, 2))[None, :, None]+1e-9)
                z = conv - b_
                feats[:, 2*sub] = (z > 0).mean(2); feats[:, 2*sub+1] = z.max(2)
    return feats

print("Loading and building pooled features...", flush=True)
Xs, ys, subj_id = [], [], []
for i, f in enumerate(FILES):
    Xall, ph = load(f)
    k = np.isin(ph, PHON)
    Xs.append(Xall[k]); ys.append(np.array([PHON.index(v) for v in ph[k]]))
    subj_id.append(np.full(k.sum(), i))
y = np.concatenate(ys); sid = np.concatenate(subj_id)
Xcat = np.concatenate(Xs, 0)
print("pooled: %d trials, %d subjects" % (len(y), len(FILES)), flush=True)

pc = PCA(NPCA).fit(Xcat.transpose(0, 2, 1).reshape(-1, Xcat.shape[1]))
Xp = np.stack([pc.transform(z.T).T for z in Xcat], 0)
T = Xp.shape[2]; ed = np.linspace(0, T, K_WIN+1).astype(int)
COVS = []
for i in range(K_WIN):
    C = Covariances("oas").fit_transform(Xp[:, :, ed[i]:ed[i+1]])
    Cal = np.empty_like(C)
    for s in range(len(FILES)):
        m = sid == s
        W = np.real(fmp(mean_riemann(C[m]), -0.5))
        Cal[m] = np.stack([W @ c @ W for c in C[m]], 0)
    COVS.append(Cal)
print("covariances + per-subject recentering done", flush=True)

pc12 = PCA(12).fit(Xcat.transpose(0, 2, 1).reshape(-1, Xcat.shape[1]))
Xr = np.stack([pc12.transform(z.T).T for z in Xcat], 0)
RKS = {k: rocket_fast(Xr, k) for k in [500, 3000]}
for k in RKS:
    for s in range(len(FILES)):
        m = sid == s
        RKS[k][m] = StandardScaler().fit_transform(RKS[k][m])
print("rocket features done", flush=True)

def ev(Ftr, ytr, Fte, yte):
    c = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-2, 4, 8))) \
        if Ftr.shape[1] > 2500 else make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

def tangent(tr, te):
    fa, fb = [], []
    for C in COVS:
        ts = TangentSpace(metric="riemann").fit(C[tr])
        fa.append(ts.transform(C[tr])); fb.append(ts.transform(C[te]))
    return np.hstack(fa), np.hstack(fb)

strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid, y)])
NAMES = ["POOLED_riem", "POOLED_RK500", "POOLED_RK3000", "POOLED_riem+RK3000", "PERSUBJ_riem"]
scores = {k: [] for k in NAMES}
print("\nrunning 5-fold (trial-level, subject-stratified)...", flush=True)
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xcat, strat)):
    Rtr, Rte = tangent(tr, te)
    scores["POOLED_riem"].append(ev(Rtr, y[tr], Rte, y[te]))
    scores["POOLED_RK500"].append(ev(RKS[500][tr], y[tr], RKS[500][te], y[te]))
    scores["POOLED_RK3000"].append(ev(RKS[3000][tr], y[tr], RKS[3000][te], y[te]))
    s1 = StandardScaler().fit(Rtr); s2 = StandardScaler().fit(RKS[3000][tr])
    a = np.hstack([s1.transform(Rtr), s2.transform(RKS[3000][tr])])
    b = np.hstack([s1.transform(Rte), s2.transform(RKS[3000][te])])
    scores["POOLED_riem+RK3000"].append(ev(a, y[tr], b, y[te]))
    per = []
    for s in range(len(FILES)):
        mtr = (sid[tr] == s); mte = (sid[te] == s)
        if mte.sum() < 11 or mtr.sum() < 22: continue
        per.append(ev(Rtr[mtr], y[tr][mtr], Rte[mte], y[te][mte]))
    scores["PERSUBJ_riem"].append(float(np.mean(per)))
    print("  fold %d: pooled_riem %.3f | RK500 %.3f RK3000 %.3f | riem+RK %.3f | per-subj %.3f" % (
        fold, scores["POOLED_riem"][-1], scores["POOLED_RK500"][-1], scores["POOLED_RK3000"][-1],
        scores["POOLED_riem+RK3000"][-1], scores["PERSUBJ_riem"][-1]), flush=True)

print("\n=== GROUP (11-phoneme, chance 0.091) ===")
base = np.array(scores["PERSUBJ_riem"])
for k in NAMES:
    a = np.array(scores[k]); _, p = ttest_rel(a, base)
    print("  %-20s %.3f +- %.3f   vs per-subject %+.3f (p=%.4f) %s" % (
        k, a.mean(), a.std(), a.mean()-base.mean(), p, "**BEATS**" if p < 0.05 and a.mean() > base.mean() else ""))
