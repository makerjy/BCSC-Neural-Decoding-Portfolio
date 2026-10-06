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


def _class_patterns(Z, y, n_cls, sub=6):
    """Per-subject class-mean spatiotemporal patterns, (d, n_cls * T_sub)."""
    P = [Z[y == c].mean(0)[:, ::sub] for c in range(n_cls)]
    return np.concatenate(P, axis=1)


def fit_alignment(Zs, ys, n_cls, n_iter=6):
    """Iterative Procrustes onto a group template.

    Each subject gets an orthogonal W_i mapping its own space into one common space, so the
    big model downstream sees aligned signals. Subject differences here are rotations, not
    offsets -- centring, scaling, indicators and scatter correction all failed to remove them
    -- and a rotation has to be undone at the front, not after the representation is built.

    LEAKAGE GUARD: patterns and template come from training trials only. Test trials are never
    used to fit W_i; they are only transformed by it.
    """
    pats = [_class_patterns(Z, y, n_cls) for Z, y in zip(Zs, ys)]
    tmpl = np.mean(pats, axis=0)
    W = [np.eye(p.shape[0]) for p in pats]
    for _ in range(n_iter):
        for i, p in enumerate(pats):
            U, _, Vt = np.linalg.svd(p @ tmpl.T)
            W[i] = (U @ Vt).T
        tmpl = np.mean([W[i] @ p for i, p in enumerate(pats)], axis=0)
    return W


def _lda_within_subject(Ftr, ytr, str_, Fte, n_cls):
    d = Ftr.shape[1]
    Sw = np.zeros((d, d)); n_cell = 0
    for s_ in np.unique(str_):
        for c in range(n_cls):
            m = (str_ == s_) & (ytr == c)
            if m.sum() < 3: continue
            Z = Ftr[m] - Ftr[m].mean(0)
            Sw += Z.T @ Z; n_cell += m.sum() - 1
    Sw /= max(n_cell, 1)
    lam = 0.3
    Sw = (1 - lam) * Sw + lam * np.trace(Sw) / d * np.eye(d)
    mu = np.stack([Ftr[ytr == c].mean(0) for c in range(n_cls)])
    Wc = np.linalg.solve(Sw + 1e-8 * np.eye(d), mu.T).T
    b = -0.5 * np.einsum("cd,cd->c", mu, Wc)
    return np.argmax(Fte @ Wc.T + b, axis=1)


def _big_model(Zal, y, sid, tr, te, K, tau, n_cls):
    """The big model, unchanged: unified delay-augmented SPD manifold + window features,
    pooled across subjects, one shared classifier."""
    C = unified_cov(Zal, 2, tau)
    Cr = recentre(C, sid, tr)
    ts = TangentSpace(metric="logeuclid").fit(Cr[tr])
    F = ts.transform(Cr)
    ed = np.linspace(0, Zal.shape[2], K + 1).astype(int)
    tw, mw = [], []
    for i in range(K):
        seg = Zal[:, :, ed[i]:ed[i + 1]]
        Cw = Covariances("oas").fit_transform(seg)
        t2 = TangentSpace(metric="logeuclid").fit(Cw[tr])
        tw.append(t2.transform(Cw)); mw.append(seg.mean(2))
    G = np.hstack([x.reshape(len(x), -1) for x in tw + mw])
    sc1 = StandardScaler().fit(G[tr]); sc2 = StandardScaler().fit(F[tr])
    A = np.hstack([sc1.transform(G), sc2.transform(F)])
    return clf(A[tr], y[tr], A[te], y[te])


def _one_fold(args):
    (X, y, sid, tr, te, K, tau, n_sub, sd, fi) = args
    n_cls = len(np.unique(y))
    pc = PCA(20).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)
    out = {}
    out["pooled_noalign"] = _big_model(Z, y, sid, tr, te, K, tau, n_cls)

    Zs = [Z[tr][sid[tr] == s_] for s_ in range(n_sub)]
    ys = [y[tr][sid[tr] == s_] for s_ in range(n_sub)]
    Wl = fit_alignment(Zs, ys, n_cls)
    Zal = Z.copy()
    for s_ in range(n_sub):
        m = sid == s_
        Zal[m] = np.einsum("ij,njt->nit", Wl[s_], Z[m])
    out["pooled_ALIGN"] = _big_model(Zal, y, sid, tr, te, K, tau, n_cls)

    Zs2 = [Z[sid == s_] for s_ in range(n_sub)]
    ys2 = [y[sid == s_] for s_ in range(n_sub)]
    Wl2 = fit_alignment(Zs2, ys2, n_cls)
    Zlk = Z.copy()
    for s_ in range(n_sub):
        m = sid == s_
        Zlk[m] = np.einsum("ij,njt->nit", Wl2[s_], Z[m])
    out["align_LEAKY_check"] = _big_model(Zlk, y, sid, tr, te, K, tau, n_cls)

    per = []
    for s_ in range(n_sub):
        mtr, mte = sid[tr] == s_, sid[te] == s_
        if mte.sum() < 11 or mtr.sum() < 33: continue
        idx = np.concatenate([tr[mtr], te[mte]])
        rel_tr = np.arange(mtr.sum()); rel_te = np.arange(mtr.sum(), len(idx))
        per.append(_big_model(Z[idx], y[idx], np.zeros(len(idx), int),
                              rel_tr, rel_te, K, tau, n_cls))
    out["per_subject_samefolds"] = float(np.mean(per))
    return sd, fi, out


def main(task, seeds, tau, data_dir, workers=10):
    X, y, sid, n_sub = load_pooled(task, data_dir)
    K = 2 if task == "place" else 6
    chance = 0.5 if task == "place" else 1 / 11
    print("task=%s pooled trials=%d subjects=%d per-class=%.0f workers=%d" % (
        task, len(X), n_sub, len(X) / len(np.unique(y)), workers), flush=True)
    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid, y)])
    NAMES = ["per_subject_samefolds", "pooled_noalign", "pooled_ALIGN", "align_LEAKY_check"]
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
