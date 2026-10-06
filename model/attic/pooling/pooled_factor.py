"""
Pooled unified space-time manifold.

Within subjects the unified delay-augmented SPD matrix already beats separate features
(+0.024, p=0.003). This asks the question that actually matters for the thesis: with all
subjects on one manifold, does a richer structure become affordable and produce a jump rather
than a creep?

Mechanism to attack: shared phoneme geometry exists at group level (RDM correlation +0.226,
p<0.0005) but is buried in per-subject estimation noise (split-half reliability 0.432).
Pooling gives ~7000 trials instead of ~460, which reduces exactly that noise. And structures
unaffordable per subject (d=40, p=4) become affordable.

Naive pooling failed before (0.121 vs 0.174) because subject baselines dominate. The fix here
is Riemannian recentring: transport each subject's geometric mean to the identity, which is
the principled group action on this manifold, estimated on training folds only.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.linalg import fractional_matrix_power as fmp
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

DATA = "npz2"
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
WMIN, WMAX = 0.0, 1.01
PER_SUBJECT_REF = 0.270  # measured in unified2b: cov+mean+unified, 16 subjects
BASE = (-0.2, -0.11)


def load_subject(path):
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme1"]})
    keep = ((meta.task == "single_phoneme") & (meta.tms_target == "control")).values
    bm = (t >= BASE[0]) & (t <= BASE[1]); wm = (t >= WMIN) & (t <= WMAX)
    Xk = X[keep]
    base = Xk[:, :, bm].mean(2, keepdims=True) if bm.any() else 0.0
    return (Xk - base)[:, :, wm], meta.phoneme1.values[keep]


def load_pooled(task, data_dir):
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "*.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    Xs, ys, sid = [], [], []
    for i, f in enumerate(files):
        Xb, ph = load_subject(f)
        m = np.isin(ph, STOPS) if task == "place" else np.isin(ph, PHON)
        y = (np.where(np.isin(ph[m], ["d", "t"]), 1, 0) if task == "place"
             else np.array([PHON.index(v) for v in ph[m]]))
        Xs.append(Xb[m]); ys.append(y); sid.append(np.full(m.sum(), i))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sid), len(files)


def unified_cov(A, p, tau, shrink=0.08):
    T = A.shape[2]
    if p > 1:
        span = (p - 1) * tau
        A = np.concatenate([A[:, :, i * tau:T - span + i * tau] for i in range(p)], 1)
    Ac = A - A.mean(2, keepdims=True)
    C = np.einsum("nit,njt->nij", Ac, Ac) / Ac.shape[2]
    k = C.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / k
    return (1 - shrink) * C + shrink * tr * np.eye(k)


def _logm(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _expm_half_inv(S):
    w, V = np.linalg.eigh(S)
    return (V * np.exp(-0.5 * w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def recentre(C, sid, tr_idx):
    """Log-Euclidean recentring: reference = exp(mean(log C)) per subject, whitened to identity.
    Closed form, so it does not need the iterative Karcher mean, which dominated runtime at
    3328 trials x 81x81 matrices per fold."""
    out = np.empty_like(C)
    L = _logm(C)
    for s in np.unique(sid):
        tr_s = tr_idx[sid[tr_idx] == s]
        m = sid == s
        if len(tr_s) < 10:
            out[m] = C[m]; continue
        W = _expm_half_inv(L[tr_s].mean(0))
        out[m] = np.einsum("ij,njk,kl->nil", W, C[m], W)
    return out


def window_parts(Xtr, Xte, K, d):
    pc = PCA(min(d, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    f = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = f(Xtr), f(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    ta, tb, ma, mb = [], [], [], []
    for i in range(K):
        sa, sb = A[:, :, ed[i]:ed[i+1]], B[:, :, ed[i]:ed[i+1]]
        Ca = Covariances("oas").fit_transform(sa); Cb = Covariances("oas").fit_transform(sb)
        ts = TangentSpace(metric="riemann").fit(Ca)
        ta.append(ts.transform(Ca)); tb.append(ts.transform(Cb))
        ma.append(sa.mean(2)); mb.append(sb.mean(2))
    g = lambda L: np.hstack([x.reshape(len(x), -1) for x in L])
    return g(ta), g(tb), g(ma), g(mb)


def clf(Ftr, ytr, Fte, yte):
    est = (make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-2, 4, 8)))
           if Ftr.shape[1] > 3000
           else make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")))
    est.fit(Ftr, ytr)
    return balanced_accuracy_score(yte, est.predict(Fte))


def stack(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


def _lda_within_subject(Ftr, ytr, str_, Fte, n_cls):
    """LDA whose within-class scatter is pooled WITHIN (subject, class).

    Standard LDA pools within-class scatter across subjects, so subject variance inflates it
    and the discriminant directions bend toward whatever separates people. Estimating scatter
    inside each (subject, class) cell removes subject variance from the denominator, which is
    the discriminant-level version of factorising subject out of phoneme.
    """
    d = Ftr.shape[1]
    Sw = np.zeros((d, d))
    n_cell = 0
    for s_ in np.unique(str_):
        for c in range(n_cls):
            m = (str_ == s_) & (ytr == c)
            if m.sum() < 3: continue
            Z = Ftr[m] - Ftr[m].mean(0)
            Sw += Z.T @ Z
            n_cell += m.sum() - 1
    Sw /= max(n_cell, 1)
    lam = 0.3
    Sw = (1 - lam) * Sw + lam * np.trace(Sw) / d * np.eye(d)
    mu = np.stack([Ftr[ytr == c].mean(0) for c in range(n_cls)])
    Wc = np.linalg.solve(Sw + 1e-8 * np.eye(d), mu.T).T
    b = -0.5 * np.einsum("cd,cd->c", mu, Wc)
    return np.argmax(Fte @ Wc.T + b, axis=1)


def _subject_center(F, sid, tr, scale=False):
    out = F.copy()
    for s_ in np.unique(sid):
        m = sid == s_
        mtr = m[tr] if len(tr) == len(sid) else (sid[tr] == s_)
        ref = F[tr][mtr]
        if len(ref) < 5: continue
        out[m] = F[m] - ref.mean(0)
        if scale:
            out[m] = out[m] / (ref.std(0) + 1e-8)
    return out


def _one_fold(args):
    (X, y, sid, tr, te, K, tau, n_sub, sd, fi) = args
    n_cls = len(np.unique(y))
    pc = PCA(20).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
    A = np.stack([pc.transform(z.T).T for z in X], 0)
    C20 = unified_cov(A, 2, tau)
    Cr = recentre(C20, sid, tr)
    ts = TangentSpace(metric="logeuclid").fit(Cr[tr])
    Fr = ts.transform(Cr)
    wa, wb, mua, mub = window_parts(X[tr], X[te], K, 20)
    a, b = stack([wa, mua, Fr[tr]], [wb, mub, Fr[te]])
    full = np.zeros((len(X), a.shape[1]))
    full[tr], full[te] = a, b

    out = {}
    out["pooled"] = clf(a, y[tr], b, y[te])
    Fc = _subject_center(full, sid, tr)
    out["+subj_center"] = clf(Fc[tr], y[tr], Fc[te], y[te])
    Fz = _subject_center(full, sid, tr, scale=True)
    out["+subj_zscore"] = clf(Fz[tr], y[tr], Fz[te], y[te])
    oh = np.eye(n_sub)[sid]
    a2 = np.hstack([Fc[tr], oh[tr]]); b2 = np.hstack([Fc[te], oh[te]])
    out["+subj_center+onehot"] = clf(a2, y[tr], b2, y[te])
    pred = _lda_within_subject(Fc[tr], y[tr], sid[tr], Fc[te], n_cls)
    out["+subj_center+withinLDA"] = balanced_accuracy_score(y[te], pred)

    per = []
    for s_ in range(n_sub):
        mtr, mte = sid[tr] == s_, sid[te] == s_
        if mte.sum() < 11 or mtr.sum() < 33: continue
        Xs, Xq = X[tr][mtr], X[te][mte]
        ws, wq, ms, mq = window_parts(Xs, Xq, K, 20)
        pcs = PCA(20).fit(Xs.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        As = np.stack([pcs.transform(z.T).T for z in Xs], 0)
        Aq = np.stack([pcs.transform(z.T).T for z in Xq], 0)
        Cs, Cq = unified_cov(As, 2, tau), unified_cov(Aq, 2, tau)
        t2 = TangentSpace(metric="logeuclid").fit(Cs)
        aa, bb = stack([ws, ms, t2.transform(Cs)], [wq, mq, t2.transform(Cq)])
        per.append(clf(aa, y[tr][mtr], bb, y[te][mte]))
    out["per_subject_samefolds"] = float(np.mean(per))
    return sd, fi, out


def main(task, seeds, tau, data_dir, workers=10):
    X, y, sid, n_sub = load_pooled(task, data_dir)
    K = 2 if task == "place" else 6
    chance = 0.5 if task == "place" else 1 / 11
    print("task=%s pooled trials=%d subjects=%d per-class=%.0f workers=%d" % (
        task, len(X), n_sub, len(X) / len(np.unique(y)), workers), flush=True)
    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid, y)])
    NAMES = ["per_subject_samefolds", "pooled", "+subj_center", "+subj_zscore",
             "+subj_center+onehot", "+subj_center+withinLDA"]
    jobs = []
    for sd in range(seeds):
        for fi, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat)):
            jobs.append((X, y, sid, tr, te, K, tau, n_sub, sd, fi))
    R = {k: [] for k in NAMES}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sd, fi, out in ex.map(_one_fold, jobs):
            for k in NAMES: R[k].append(out[k])
            print("  seed %d fold %d  %s" % (sd, fi, "  ".join(
                "%s %.3f" % (k[:14], out[k]) for k in NAMES)), flush=True)
    print(chr(10) + "=== %s (chance %.3f) ===" % (task, chance))
    base = np.array(R["per_subject_samefolds"])
    for k in NAMES:
        a = np.array(R[k]); _, p0 = ttest_1samp(a, chance); _, pb = ttest_rel(a, base)
        print("  %-22s %.3f  vs-chance p=%.7f  vs-per-subject %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean() - base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(a.task, a.seeds, a.tau, a.data, a.workers)
