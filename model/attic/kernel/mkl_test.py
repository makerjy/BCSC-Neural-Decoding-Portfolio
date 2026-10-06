import warnings
warnings.filterwarnings("ignore")
import glob
import os
import time
import numpy as np
import pandas as pd
from scipy.linalg import logm
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5

def dda(x, tau1=7, tau2=10):
    n, c, T = x.shape; mm = T-tau2-1; o = np.empty((n, c, 4), np.float32)
    for i in range(n):
        for j in range(c):
            u = x[i, j]; ud = u[tau2+1:tau2+1+mm]-u[tau2:tau2+mm]
            a = u[tau2+1-tau1:tau2+1-tau1+mm]; bb = u[1:1+mm]; D = np.stack([a, bb, a*a], 1)
            cf, *_ = np.linalg.lstsq(D, ud, rcond=None); o[i, j] = [cf[0], cf[1], cf[2], float(np.mean((D@cf-ud)**2))]
    return o.reshape(n, -1)

def spd(x, k=15):
    n, c, T = x.shape; p = PCA(k).fit(x.transpose(0, 2, 1).reshape(-1, c))
    xp = np.stack([p.transform(x[i].T).T for i in range(n)], 0)
    iu = np.triu_indices(k); off = iu[0] != iu[1]; F = np.empty((n, len(iu[0])), np.float32)
    for i in range(n):
        s = xp[i]; cov = (s@s.T)/(s.shape[1]-1); cov = 0.85*cov+0.15*np.trace(cov)/k*np.eye(k)
        L = logm(cov).real; v = L[iu].copy(); v[off] *= np.sqrt(2); F[i] = v
    return F

def K(F):
    Fs = StandardScaler().fit_transform(F)
    return rbf_kernel(Fs, gamma=1.0/(Fs.shape[1]*Fs.var()))

def center(Kt):
    n = Kt.shape[0]; H = np.eye(n)-np.ones((n, n))/n
    return H@Kt@H

def kta_weights(Ks_tr, y_tr):
    yb = np.where(y_tr == 1, 1.0, -1.0); Ty = np.outer(yb, yb); Tc = center(Ty); tn = np.linalg.norm(Tc)
    w = []
    for Kt in Ks_tr:
        Kc = center(Kt); a = np.sum(Kc*Tc)/(np.linalg.norm(Kc)*tn+1e-12); w.append(max(a, 0.0))
    w = np.array(w); return w/w.sum() if w.sum() > 0 else np.ones(len(w))/len(w)

def cv_combiner(Klist, y, seed, mode):
    a, ws = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        if mode == "add":
            w = np.ones(len(Klist))/len(Klist)
        else:
            w = kta_weights([Kk[np.ix_(tr, tr)] for Kk in Klist], y[tr])
        ws.append(w)
        Ktr = sum(w[i]*Klist[i][np.ix_(tr, tr)] for i in range(len(Klist)))
        Kte = sum(w[i]*Klist[i][np.ix_(te, tr)] for i in range(len(Klist)))
        clf = SVC(kernel="precomputed", class_weight="balanced"); clf.fit(Ktr, y[tr])
        a.append(balanced_accuracy_score(y[te], clf.predict(Kte)))
    return float(np.mean(a)), np.mean(ws, 0)

def cv_single(Kt, y, seed):
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", class_weight="balanced"); clf.fit(Kt[np.ix_(tr, tr)], y[tr])
        a.append(balanced_accuracy_score(y[te], clf.predict(Kt[np.ix_(te, tr)])))
    return float(np.mean(a))

R = {c: {"V": [], "ADD": [], "MKL": [], "W": []} for c in ["cat", "pl"]}
t0 = time.time()
print("MKL (kernel-target-alignment weights) vs additive vs voltage | %d subj\n" % len(FILES))
print("%-7s | place: V  ADD  MKL  | MKL weights [V,D,S]" % "subj")
for idx, f in enumerate(FILES, 1):
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Kv, Kd, Ksp = K(Xb[:, :, ::10].reshape(len(Xb), -1)), K(dda(Xb)), K(spd(Xb))
    ph = ds.phoneme.values
    for cond, mask, yv in [
        ("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
        ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]; Kl = [Kv[np.ix_(mask, mask)], Kd[np.ix_(mask, mask)], Ksp[np.ix_(mask, mask)]]
        v = np.mean([cv_single(Kl[0], yy, s) for s in range(5)])
        adds = [cv_combiner(Kl, yy, s, "add") for s in range(5)]
        mkls = [cv_combiner(Kl, yy, s, "mkl") for s in range(5)]
        R[cond]["V"].append(v); R[cond]["ADD"].append(np.mean([x[0] for x in adds]))
        R[cond]["MKL"].append(np.mean([x[0] for x in mkls])); R[cond]["W"].append(np.mean([x[1] for x in mkls], 0))
    wpl = R["pl"]["W"][-1]
    print("%-7s | %.3f %.3f %.3f | [%.2f %.2f %.2f]  [%d/%d %.1fm]" % (
        sub, R["pl"]["V"][-1], R["pl"]["ADD"][-1], R["pl"]["MKL"][-1], wpl[0], wpl[1], wpl[2],
        idx, len(FILES), (time.time()-t0)/60), flush=True)

print("\n=== GROUP (paired-t) ===")
for cond, lbl in [("cat", "category"), ("pl", "place-stops")]:
    V = np.array(R[cond]["V"]); A = np.array(R[cond]["ADD"]); M = np.array(R[cond]["MKL"])
    _, pv = ttest_1samp(V, .5); _, pm = ttest_1samp(M, .5)
    _, pmv = ttest_rel(M, V); _, pma = ttest_rel(M, A)
    W = np.array(R[cond]["W"]).mean(0)
    print("  %-12s V %.3f  ADD %.3f  MKL %.3f(p_chance=%.3f) | MKL-V=%+.3f(p=%.3f) MKL-ADD=%+.3f(p=%.3f) | meanW[V,D,S]=[%.2f,%.2f,%.2f]" % (
        lbl, V.mean(), A.mean(), M.mean(), pm, M.mean()-V.mean(), pmv, M.mean()-A.mean(), pma, W[0], W[1], W[2]))
