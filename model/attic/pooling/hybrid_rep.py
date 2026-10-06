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
K_WIN, NPCA, NK = 6, 20, 1500

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

Xs, ys, sids = [], [], []
for i, f in enumerate(FILES):
    Xall, ph = load(f)
    k = np.isin(ph, PHON)
    Xs.append(Xall[k]); ys.append(np.array([PHON.index(v) for v in ph[k]])); sids.append(np.full(k.sum(), i))
X = np.concatenate(Xs, 0); y = np.concatenate(ys); sid = np.concatenate(sids)
NS = len(FILES)
print("pooled %d trials / %d subjects\n" % (len(y), NS), flush=True)

pc_glob = PCA(NPCA).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
Xg = np.stack([pc_glob.transform(z.T).T for z in X], 0)
T = Xg.shape[2]; ED = np.linspace(0, T, K_WIN+1).astype(int)
COV_G = [Covariances("oas").fit_transform(Xg[:, :, ED[i]:ED[i+1]]) for i in range(K_WIN)]
pc12 = PCA(12).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
RK = rocket_fast(np.stack([pc12.transform(z.T).T for z in X], 0), NK)
for s in range(NS):
    RK[sid == s] = StandardScaler().fit_transform(RK[sid == s])
print("global features ready", flush=True)

def clf(d):
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-2, 4, 8))) if d > 2500 \
        else make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def recenter_by_subject(C, idx_tr, idx_all):
    out = np.empty_like(C[idx_all])
    for s in range(NS):
        m_all = sid[idx_all] == s
        m_tr = sid[idx_tr] == s
        if m_tr.sum() < 5: continue
        W = np.real(fmp(mean_riemann(C[idx_tr][m_tr]), -0.5))
        out[m_all] = np.stack([W @ c @ W for c in C[idx_all][m_all]], 0)
    return out

NAMES = ["A_all_persubj", "B_all_pooled", "C_pooledRep_persubjClf", "D_C_plus_ROCKET", "E_shared+private"]
S = {k: [] for k in NAMES}
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, np.array(["%d_%d" % (a, b) for a, b in zip(sid, y)])))
for fi, (tr, te) in enumerate(folds):
    Rtr_g, Rte_g = [], []
    for i in range(K_WIN):
        Ctr = recenter_by_subject(COV_G[i], tr, tr)
        Cte = recenter_by_subject(COV_G[i], tr, te)
        ts = TangentSpace(metric="riemann").fit(Ctr)
        Rtr_g.append(ts.transform(Ctr)); Rte_g.append(ts.transform(Cte))
    Rtr_g, Rte_g = np.hstack(Rtr_g), np.hstack(Rte_g)

    c = clf(Rtr_g.shape[1]); c.fit(Rtr_g, y[tr])
    S["B_all_pooled"].append(balanced_accuracy_score(y[te], c.predict(Rte_g)))

    accA, accC, accD, accE = [], [], [], []
    for s in range(NS):
        mtr = sid[tr] == s; mte = sid[te] == s
        if mte.sum() < 11 or mtr.sum() < 33: continue
        Xtr_s, Xte_s = X[tr][mtr], X[te][mte]
        pc_s = PCA(NPCA).fit(Xtr_s.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        prj = lambda Z: np.stack([pc_s.transform(z.T).T for z in Z], 0)
        A_tr, A_te = prj(Xtr_s), prj(Xte_s)
        fa, fb = [], []
        for i in range(K_WIN):
            Ca = Covariances("oas").fit_transform(A_tr[:, :, ED[i]:ED[i+1]])
            Cb = Covariances("oas").fit_transform(A_te[:, :, ED[i]:ED[i+1]])
            t_ = TangentSpace(metric="riemann").fit(Ca)
            fa.append(t_.transform(Ca)); fb.append(t_.transform(Cb))
        fa, fb = np.hstack(fa), np.hstack(fb)
        c = clf(fa.shape[1]); c.fit(fa, y[tr][mtr])
        accA.append(balanced_accuracy_score(y[te][mte], c.predict(fb)))

        c = clf(Rtr_g.shape[1]); c.fit(Rtr_g[mtr], y[tr][mtr])
        accC.append(balanced_accuracy_score(y[te][mte], c.predict(Rte_g[mte])))

        s1 = StandardScaler().fit(Rtr_g[mtr]); s2 = StandardScaler().fit(RK[tr][mtr])
        a2 = np.hstack([s1.transform(Rtr_g[mtr]), s2.transform(RK[tr][mtr])])
        b2 = np.hstack([s1.transform(Rte_g[mte]), s2.transform(RK[te][mte])])
        c = clf(a2.shape[1]); c.fit(a2, y[tr][mtr])
        accD.append(balanced_accuracy_score(y[te][mte], c.predict(b2)))

        s3 = StandardScaler().fit(fa)
        a3 = np.hstack([s1.transform(Rtr_g[mtr]), s3.transform(fa)])
        b3 = np.hstack([s1.transform(Rte_g[mte]), s3.transform(fb)])
        c = clf(a3.shape[1]); c.fit(a3, y[tr][mtr])
        accE.append(balanced_accuracy_score(y[te][mte], c.predict(b3)))
    S["A_all_persubj"].append(np.mean(accA)); S["C_pooledRep_persubjClf"].append(np.mean(accC))
    S["D_C_plus_ROCKET"].append(np.mean(accD)); S["E_shared+private"].append(np.mean(accE))
    print("  fold %d: A %.3f | B %.3f | C %.3f | D %.3f | E %.3f" % (
        fi, S["A_all_persubj"][-1], S["B_all_pooled"][-1], S["C_pooledRep_persubjClf"][-1],
        S["D_C_plus_ROCKET"][-1], S["E_shared+private"][-1]), flush=True)

print("\n=== GROUP (11-phoneme, chance 0.091) ===")
base = np.array(S["A_all_persubj"])
for k in NAMES:
    a = np.array(S[k]); _, p = ttest_rel(a, base)
    print("  %-24s %.3f +- %.3f  vs A %+.3f (p=%.4f) %s" % (
        k, a.mean(), a.std(), a.mean()-base.mean(), p, "**BEATS**" if p < 0.05 and a.mean() > base.mean() else ""))
