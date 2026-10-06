import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import stft
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
WMIN, WMAX, SF = 0.0, 0.5, 500.0
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def rocket_transform(X, n_kernels=500, seed=0):
    rng = np.random.default_rng(seed)
    n, C, T = X.shape
    feats = np.empty((n, n_kernels * 2), np.float32)
    for k in range(n_kernels):
        L = rng.choice([7, 9, 11])
        ch = rng.choice(C, size=min(C, rng.choice([1, 2, 4])), replace=False)
        w = rng.normal(0, 1, (len(ch), L)); w = w - w.mean()
        max_d = max(1, int(np.log2((T - 1) / (L - 1))))
        dil = int(2 ** rng.uniform(0, max_d))
        span = (L - 1) * dil
        if span >= T:
            dil = 1; span = L - 1
        idx = np.arange(0, L) * dil
        seg = np.stack([X[:, ch, i:T - span + i] for i in idx], -1)
        conv = np.einsum("ncti,ci->nt", seg, w)
        b = rng.uniform(-1, 1) * (np.abs(conv).mean() + 1e-9)
        z = conv - b
        feats[:, 2*k] = (z > 0).mean(1)
        feats[:, 2*k+1] = z.max(1)
    return feats

def ngrc_feats(X, npca=6, k_delays=3, tau=15):
    n, C, T = X.shape
    Z = X[:, :npca, :] if C >= npca else X
    lin = np.concatenate([Z[:, :, i*tau:T - (k_delays-1)*tau + i*tau] for i in range(k_delays)], axis=1)
    lin_s = lin[:, :, ::35]
    d = lin_s.shape[1]
    iu = np.triu_indices(d)
    quad = np.einsum("ndt,net->ndet", lin_s, lin_s)[:, iu[0], iu[1], :]
    return np.concatenate([lin_s.reshape(n, -1), quad.reshape(n, -1)], 1)

def tf_feats(X, npca=12):
    f, tt, Zx = stft(X, fs=SF, nperseg=64, noverlap=48, axis=2)
    band = (f >= 1) & (f <= 60)
    mag = np.log(np.abs(Zx[:, :, band, :]) + 1e-12)
    ph = np.angle(Zx[:, :, band, :])
    return np.concatenate([mag.reshape(len(X), -1), np.cos(ph).reshape(len(X), -1)], 1)

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

def ev(Ftr, ytr, Fte, yte, kind="lda"):
    if kind == "ridge":
        c = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    else:
        c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

NAMES = ["winriem", "ROCKET", "NGRC", "TimeFreq", "winriem+ROCKET", "winriem+NGRC+ROCKET"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("ROCKET (dilated random conv + PPV) | NG-RC (delay+polynomial) | Time-Frequency (STFT+phase)\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        pcg = PCA(12).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        Xp = np.stack([pcg.transform(z.T).T for z in X], 0)
        RK = rocket_transform(Xp)
        NG = ngrc_feats(Xp)
        TF = tf_feats(Xp)
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                acc["ROCKET"].append(ev(RK[tr], y[tr], RK[te], y[te], "ridge"))
                acc["NGRC"].append(ev(NG[tr], y[tr], NG[te], y[te]))
                acc["TimeFreq"].append(ev(TF[tr], y[tr], TF[te], y[te]))
                s1 = StandardScaler().fit(Rtr); s2 = StandardScaler().fit(RK[tr])
                a = np.hstack([s1.transform(Rtr), s2.transform(RK[tr])])
                b = np.hstack([s1.transform(Rte), s2.transform(RK[te])])
                acc["winriem+ROCKET"].append(ev(a, y[tr], b, y[te]))
                s3 = StandardScaler().fit(NG[tr])
                a2 = np.hstack([a, s3.transform(NG[tr])]); b2 = np.hstack([b, s3.transform(NG[te])])
                acc["winriem+NGRC+ROCKET"].append(ev(a2, y[tr], b2, y[te]))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %s | ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k][-1] for k in NAMES),
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["winriem"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-22s %.3f  vs-chance p=%.7f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
