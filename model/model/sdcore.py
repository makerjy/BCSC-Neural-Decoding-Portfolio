
"""
Shared pieces for the npz2 models (RESULTS.md section 11).

Before this module the experiment scripts imported from each other --
nonlin -> final_model -> arch -> geom -> hier2 -> hier -- so every script was also a library
and the same six primitives were copy-pasted several times. Everything reusable now lives
here and the scripts are drivers: a `_fold` that says what to compare and a `main` that says
on which folds.

Contents
    data          load_subject, load_pooled, the task definitions
    folds         trial / token / time / loso, all returning (train_idx, test_idx) pairs
    spd           spd_log, cov, triu, delay_aug, phi_rows
    spatial       laplacian_modes, montage_laplacian, basis_full, basis_modes,
                  scatters, noise_scatter, disc_filter, snr_filter
    represent     window_blocks, unified_spd, tangent, block_features, pca_blocks
    readout       lda_pooled, lda_persubj, persubj_pipeline          plain
                  cell_means, shared_precision, personal_precision, gauss_predict, blend,
                  pick_gamma                                          hierarchical means
                  prepare, score_grid, band_scores, tune_head, head   two-parameter family
    report        vs-chance / paired table used by every script

Conventions that matter for correctness, all verified by the permutation null in
final_model.py (real 0.252, permuted 0.088 = chance):
  * everything fitted -- spatial filters, PCA, tangent references, scalers, class statistics
    -- is fitted on the training index only;
  * `tangent(adapt="unlabeled")` is the one exception and uses no labels, only each subject's
    own trials, which is legitimate for a subject with no labelled data;
  * npz2 epochs start at t=0, so there is no pre-stimulus baseline to subtract. That is why
    the window means and the temporal rows carry so much: they are the only place the evoked
    DC offset survives.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data lives at the repo root, one level above model/, so the scripts work whether you run
# them from the repo root (`python model/final_model.py`) or from inside model/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "npz2") if os.path.isdir(os.path.join(_ROOT, "npz2")) else "npz2"

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
VOWELS = {"a", "i", "e", "u", "o"}
STOPS = {"b": 0, "p": 0, "d": 1, "t": 1}
WMIN, WMAX = 0.0, 1.01
CHANCE = {"ph11": 1 / 11, "cv": 0.5, "place": 0.5}


# ---------------------------------------------------------------- data

def load_subject(path, task):
    """One subject's single_phoneme control trials, 0-1.0s. Returns X, y, stimulus token id."""
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]]
                         for c in ["task", "tms_target", "phoneme1", "stimulus"]})
    keep = ((meta.task == "single_phoneme") & (meta.tms_target == "control")).values
    Xk = X[keep][:, :, (t >= WMIN) & (t <= WMAX)]
    ph = meta.phoneme1.values[keep]
    tok = np.array([int(s[-1]) if s and s[-1].isdigit() else 0
                    for s in meta.stimulus.values[keep]])
    if task == "ph11":
        m = np.isin(ph, PHON)
        y = np.array([PHON.index(v) for v in ph[m]])
    elif task == "cv":
        m = np.isin(ph, PHON)
        y = np.array([0 if v in VOWELS else 1 for v in ph[m]])
    elif task == "place":
        m = np.isin(ph, list(STOPS))
        y = np.array([STOPS[v] for v in ph[m]])
    else:
        raise ValueError(task)
    return Xk[m], y, tok[m]


def load_pooled(task, data_dir=DATA, normalise=False):
    """All 61-channel subjects pooled. Returns X, y, tok, sid, pos, n_sub.

    normalise divides X by its global standard deviation. The reservoir gain and the
    temporal-row scaling are written against that convention and the plain covariance models
    are not, so it is left to the caller rather than defaulted.
    """
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "*.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    Xs, ys, ts, sid = [], [], [], []
    for i, f in enumerate(files):
        Xi, yi, ti = load_subject(f, task)
        Xs.append(Xi); ys.append(yi); ts.append(ti); sid.append(np.full(len(yi), i))
    X = np.concatenate(Xs)
    if normalise:
        X = X / X.std()
    pos = np.asarray(np.load(files[0], allow_pickle=True)["channel_positions_xyz"], float)
    return (X, np.concatenate(ys), np.concatenate(ts), np.concatenate(sid), pos, len(files))


# ---------------------------------------------------------------- folds

def trial_folds(X, y, sid, seeds=1, n_splits=5):
    """Stratified by (subject, class): every fold contains all subjects and all classes."""
    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    return [f for sd in range(seeds)
            for f in StratifiedKFold(n_splits, shuffle=True, random_state=sd).split(X, strat)]


def token_folds(tok, y, sid):
    """Leave one acoustic exemplar out. Each phoneme has 4-5 recordings, so a random trial
    split puts the same audio file on both sides; this does not."""
    out = []
    for g in [1, 2, 3, 5]:
        te = np.where(tok == g)[0]
        tr = np.where(tok != g)[0]
        if len(te) and len(np.unique(y[te])) == len(np.unique(y)):
            out.append((tr, te))
    return out


def time_folds(sid, n_folds=5):
    """Blocked CV in acquisition order (trials are stored in trial_n order). Random splits put
    temporally adjacent trials on both sides and this dataset drifts hard -- trial-half
    decoding 0.919 -- so this separates a stimulus-locked signal from a session trend."""
    out = []
    for f in range(n_folds):
        tr, te = [], []
        for s in np.unique(sid):
            idx = np.where(sid == s)[0]
            ed = np.linspace(0, len(idx), n_folds + 1).astype(int)
            te.append(idx[ed[f]:ed[f + 1]])
            tr.append(np.concatenate([idx[:ed[f]], idx[ed[f + 1]:]]))
        out.append((np.concatenate(tr), np.concatenate(te)))
    return out


def loso_folds(sid):
    """Leave one subject out: the test subject contributes nothing to training."""
    return [(np.where(sid != s)[0], np.where(sid == s)[0]) for s in np.unique(sid)]


def folds_for(kind, X, y, tok, sid, seeds=1):
    if kind == "trial":
        return trial_folds(X, y, sid, seeds)
    if kind == "token":
        return token_folds(tok, y, sid)
    if kind == "time":
        return time_folds(sid)
    if kind == "loso":
        return loso_folds(sid)
    raise ValueError(kind)


# ---------------------------------------------------------------- SPD primitives

def spd_log(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * np.log(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def cov(A, shrink=0.08):
    """Centred covariance with shrinkage toward a scaled identity."""
    Ac = A - A.mean(2, keepdims=True)
    C = np.einsum("nit,njt->nij", Ac, Ac) / Ac.shape[2]
    k = C.shape[-1]
    return (1 - shrink) * C + shrink * np.trace(C, axis1=1, axis2=2)[:, None, None] / k * np.eye(k)


def triu(S):
    """Upper triangle with the off-diagonal weighted by sqrt(2), so the Euclidean norm of the
    vector equals the Frobenius norm of the matrix."""
    d = S.shape[-1]
    iu = np.triu_indices(d)
    return S[..., iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def delay_aug(A, p, tau):
    """Stack p copies at lags 0, tau, ..., (p-1)tau. p=1 is a no-op."""
    if p == 1:
        return A
    T = A.shape[2]; span = (p - 1) * tau
    return np.concatenate([A[:, :, i * tau:T - span + i * tau] for i in range(p)], 1)


def phi_rows(kind, T, K, high=40, seed=0):
    """K+1 temporal basis rows at unit RMS, so every variant is matched in count and scale.

    dct_low   k = 0..K              smooth, low frequency, ordered -- the one that works
    dct_high  k = high..high+K      same smoothness class, wrong timescale
    shuffled  dct_low with its time samples permuted once
    gauss     iid rows
    none      no rows
    """
    t = (np.arange(T) + 0.5) / T
    if kind == "none":
        return np.zeros((0, T))
    if kind == "dct_low":
        P = np.stack([np.cos(np.pi * k * t) for k in range(K + 1)])
    elif kind == "dct_high":
        P = np.stack([np.cos(np.pi * k * t) for k in range(high, high + K + 1)])
    elif kind == "shuffled":
        P = np.stack([np.cos(np.pi * k * t) for k in range(K + 1)])
        P = P[:, np.random.default_rng(seed).permutation(T)]
    elif kind == "gauss":
        P = np.random.default_rng(seed).standard_normal((K + 1, T))
    else:
        raise ValueError(kind)
    return P / np.sqrt((P ** 2).mean(1, keepdims=True))


# ---------------------------------------------------------------- spatial front-ends

def laplacian_modes(pos):
    """Graph Laplacian of the montage; eigenvectors are scalp spatial frequencies, fixed by
    geometry alone so all subjects share the axes. Ordering by scale is what pays (+0.033 over
    an arbitrary rotation); the anatomy itself does not beat PCA (0.287 vs 0.281)."""
    D2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    h2 = np.median(D2[D2 > 0])
    A = np.exp(-D2 / (2 * h2))
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(1)) - A
    w, V = eigh(L)
    return V, np.clip(w, 0, None)


def montage_laplacian(pos):
    """The Laplacian operator itself, for use as a smoothness penalty. Used by the archived
    filter ablation (attic/studies/arch.py), which measured that the penalty contributes
    nothing once the between-subject nuisance term is dropped."""
    D2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    h2 = np.median(D2[D2 > 0])
    A = np.exp(-D2 / (2 * h2))
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(1)) - A


def basis_full(kind, X, tr, V, wlap, alpha=0.0, eps=0.5):
    """A full 61x61 orthonormal rotation. The four options differ only in the rotation, which
    is what makes them a fair control set for each other."""
    if kind == "lap":
        return V * ((wlap + eps) ** (-alpha / 2.0))[None, :]
    if kind == "pca":
        return PCA(X.shape[1]).fit(
            X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1])).components_.T
    if kind == "chan":
        return np.eye(X.shape[1])
    if kind == "rand":
        Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal(
            (X.shape[1], X.shape[1])))
        return Q
    raise ValueError(kind)


def basis_modes(kind, X, tr, V=None, d=14, lo=0):
    """d columns only: a Laplacian band, or the top-d principal components."""
    if kind == "lap":
        return V[:, lo:lo + d]
    return PCA(d).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1])).components_.T


def scatters(X, y, sid, tr, n_cls):
    """Spatial scatter of the mean waveforms, training trials only.
    S_phon within-subject between-class (the signal), S_subj between-subject."""
    C = X.shape[1]
    Sp = np.zeros((C, C)); Ss = np.zeros((C, C))
    Ms = []
    for s in np.unique(sid[tr]):
        i = tr[sid[tr] == s]
        Mc = [X[i[y[i] == c]].mean(0) for c in range(n_cls) if (y[i] == c).any()]
        if len(Mc) < 2:
            continue
        Mc = np.stack(Mc)
        M_s = Mc.mean(0)
        for m in Mc:
            D = m - M_s
            Sp += D @ D.T
        Ms.append(M_s)
    M = np.mean(Ms, axis=0)
    for m in Ms:
        D = m - M
        Ss += D @ D.T
    return Sp, Ss


def noise_scatter(X, y, sid, tr, n_cls):
    """Within-(subject, class) residual spatial scatter: the noise a filter divides by."""
    C = X.shape[1]
    S = np.zeros((C, C))
    for s in np.unique(sid[tr]):
        i = tr[sid[tr] == s]
        for c in range(n_cls):
            j = i[y[i] == c]
            if len(j) < 2:
                continue
            R = X[j] - X[j].mean(0)
            S += np.einsum("nit,njt->ij", R, R)
    return S


def trace_norm(A):
    t = np.trace(A)
    return A / t if t > 1e-12 else A


def disc_filter(X, y, sid, tr, n_cls, V, L, d, lo, nuis, alpha, eps, kind="disc", seed=0):
    """Spatial projection, 61 -> d, with the ablation knobs the filter study needed.

    kind="disc"  generalised eigenvectors of S_phon against nuis*S_subj + alpha*L + eps*I.
                 nuis suppresses between-subject variance, alpha penalises spatially rough
                 filters. Both measured to contribute nothing (nuis -0.002 p=0.73, and alone
                 -0.025 p=0.0002; alpha only repairs what nuis breaks), so production uses
                 snr_filter instead.
    kind="lap_band" / "pca" / "rand"  fixed-basis and capacity controls.
    """
    C = X.shape[1]
    if kind == "lap_band":
        return V[:, lo:lo + d]
    if kind == "pca":
        return PCA(d).fit(X[tr].transpose(0, 2, 1).reshape(-1, C)).components_.T
    if kind == "rand":
        Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((C, C)))
        return Q[:, :d]
    if kind != "disc":
        raise ValueError(kind)
    Sp, Ss = scatters(X, y, sid, tr, n_cls)
    den = nuis * trace_norm(Ss) + alpha * trace_norm(L) + eps * np.eye(C) / C
    w, U = eigh(trace_norm(Sp), den)
    return U[:, np.argsort(w)[::-1][:d]]


def snr_filter(X, y, sid, tr, n_cls, d, eps=0.05, den_kind="noise"):
    """Production stage 1: maximise signal-to-noise, not signal. den_kind="eye" is the
    signal-only version and loses (0.235 against 0.252)."""
    Sp, _ = scatters(X, y, sid, tr, n_cls)
    C = X.shape[1]
    den = eps * np.eye(C) / C
    if den_kind == "noise":
        den = den + trace_norm(noise_scatter(X, y, sid, tr, n_cls))
    w, U = eigh(trace_norm(Sp), den)
    return U[:, np.argsort(w)[::-1][:d]]


# ---------------------------------------------------------------- representations

def window_blocks(Z, tr, K, p, tau):
    """Separate-then-concatenate: delay-augmented tangent, then per-window tangent and mean.
    Time order lives in the window boundaries rather than inside the object."""
    T = Z.shape[2]
    blocks = []
    L = spd_log(cov(delay_aug(Z, p, tau)))
    blocks.append(triu(L - L[tr].mean(0)))
    ed = np.linspace(0, T, K + 1).astype(int)
    for i in range(K):
        seg = Z[:, :, ed[i]:ed[i + 1]]
        Lw = spd_log(cov(seg))
        blocks.append(triu(Lw - Lw[tr].mean(0)))
        blocks.append(seg.mean(2))
    return blocks


def unified_spd(Z, tr, p, tau, K, phi_scale=1.0, phi_kind="dct_low", high=40,
                shrink=0.08, seed=0):
    """One SPD matrix per trial:  A = [Z(t); Z(t-tau); ...; phi_0..phi_K],  C = A A^T / T.

    Not centred, so the temporal rows keep the evoked waveform. The Z-phi blocks are the
    projection of each mode's waveform onto temporal basis k, which is how time order gets
    inside an object that otherwise averages over t. phi_kind is the control axis: at matched
    row count dct_low beats shuffled (-0.054), gauss (-0.066), dct_high (-0.101) and no rows
    at all (-0.052), all p<1e-4.
    """
    A = delay_aug(Z, p, tau)
    n, k, T = A.shape
    P = phi_rows(phi_kind, T, K, high, seed)
    if len(P):
        P = P * (phi_scale * np.sqrt((A[tr] ** 2).mean()))
        A = np.concatenate([A, np.broadcast_to(P, (n,) + P.shape)], 1)
        k = A.shape[1]
        C = np.einsum("nit,njt->nij", A, A) / T
    else:
        Ac = A - A.mean(2, keepdims=True)
        C = np.einsum("nit,njt->nij", Ac, Ac) / T
    tc = np.trace(C, axis1=1, axis2=2)[:, None, None] / k
    return (1 - shrink) * C + shrink * tc * np.eye(k)


def tangent(C, tr, sid, adapt="train"):
    """log-Euclidean tangent, scaling, and subject centring.

    adapt="train"      reference and centre from training trials. A held-out subject has none,
                       so it enters uncentred while training subjects were centred -- an
                       artefact of the split worth +0.017 (p=0.068), not the whole LOSO gap.
    adapt="unlabeled"  reference and centre from each subject's own trials. Labels are never
                       used, so it is legitimate for a new person and implementable online as
                       a running mean (Riemannian recentring, Zanini et al. 2018).
    """
    L = spd_log(C)
    if adapt == "unlabeled":
        ref = np.empty_like(L)
        for s in np.unique(sid):
            m = sid == s
            ref[m] = L[m].mean(0)
        F = triu(L - ref)
    else:
        F = triu(L - L[tr].mean(0))
    F = StandardScaler().fit(F[tr]).transform(F)
    for s in np.unique(sid):
        m = sid == s
        src = np.where(m)[0] if adapt == "unlabeled" else tr[sid[tr] == s]
        if len(src) >= 10:
            F[m] -= F[src].mean(0)
    return F


def block_features(blocks, tr, sid, subject_center=True):
    """Standardise each block on training trials, concatenate, then centre per subject."""
    F = np.hstack([StandardScaler().fit(b[tr]).transform(b) for b in blocks])
    if subject_center:
        for s in np.unique(sid):
            tr_s = tr[sid[tr] == s]
            if len(tr_s) >= 10:
                F[sid == s] -= F[tr_s].mean(0)
    return F


def pca_blocks(X, tr, d, K, p, tau, sid=None, subject_center=False):
    """PCA front-end plus window_blocks -- the pipeline the session started from."""
    if sid is None:
        sid = np.zeros(len(X), int)
    pc = PCA(d).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)     # transform, not components_ @ X:
    return block_features(window_blocks(Z, tr, K, p, tau),  # keeps the PCA mean subtraction
                          tr, sid, subject_center)          # the original build_features had


# ---------------------------------------------------------------- plain readouts

def lda_pooled(F, y, tr, te):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(F[tr], y[tr])
    return balanced_accuracy_score(y[te], m.predict(F[te]))


def lda_persubj(F, y, sid, tr, te, min_te, min_tr):
    """One classifier per subject on the shared feature space. The measured reason: subjects'
    discriminant directions are near-orthogonal (cos 0.01 against a within-subject split-half
    of 0.45), so a single decision rule is a compromise that fits nobody."""
    per = []
    for s in np.unique(sid):
        mtr = tr[sid[tr] == s]; mte = te[sid[te] == s]
        if len(mte) < min_te or len(mtr) < min_tr:
            continue
        per.append(lda_pooled(F, y, mtr, mte))
    return float(np.mean(per)) if per else np.nan


def persubj_pipeline(X, y, sid, tr, te, d, K, p, tau, min_te, min_tr):
    """Everything personal: basis, tangent reference, scaling and classifier."""
    per = []
    for s in np.unique(sid):
        mtr = tr[sid[tr] == s]; mte = te[sid[te] == s]
        if len(mte) < min_te or len(mtr) < min_tr:
            continue
        idx = np.concatenate([mtr, mte])
        rt = np.arange(len(mtr)); rs = np.arange(len(mtr), len(idx))
        F = pca_blocks(X[idx], rt, d, K, p, tau)
        per.append(lda_pooled(F, y[idx], rt, rs))
    return float(np.mean(per)) if per else np.nan


# ---------------------------------------------------------------- hierarchical means

def cell_means(Z, y, sid, idx, n_cls, n_sub):
    """Per-(subject, class) means, the group mean, and counts. Empty cells fall back to the
    group. The group mean is near zero once features are subject-centred, which is why
    shrinking toward it turned out to be indistinguishable from shrinking toward the origin."""
    D = Z.shape[1]
    M = np.zeros((n_sub, n_cls, D)); N = np.zeros((n_sub, n_cls))
    for s in range(n_sub):
        zs = idx[sid[idx] == s]
        if not len(zs):
            continue
        for c in range(n_cls):
            m = zs[y[zs] == c]
            N[s, c] = len(m)
            if len(m):
                M[s, c] = Z[m].mean(0)
    Mp = np.zeros((n_cls, D))
    for c in range(n_cls):
        w = N[:, c] > 0
        Mp[c] = M[w, c].mean(0) if w.any() else 0.0
    for s in range(n_sub):
        for c in range(n_cls):
            if N[s, c] == 0:
                M[s, c] = Mp[c]
    return M, N, Mp


def shared_precision(Z, y, sid, idx, M):
    R = Z[idx] - M[sid[idx], y[idx]]
    S, _ = ledoit_wolf(R, assume_centered=True)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, max(w.max(), 1e-12) * 1e-10, None)
    return (V / w) @ V.T


def personal_precision(Z, y, sid, idx, M, n_sub, D):
    P = np.zeros((n_sub, D, D))
    for s in range(n_sub):
        zs = idx[sid[idx] == s]
        if len(zs) < 20:
            continue
        R = Z[zs] - M[sid[zs], y[zs]]
        S, _ = ledoit_wolf(R, assume_centered=True)
        w, V = np.linalg.eigh(S)
        w = np.clip(w, max(w.max(), 1e-12) * 1e-10, None)
        P[s] = (V / w) @ V.T
    return P


def blend(M, Mp, g):
    return g * Mp[None, :, :] + (1 - g) * M


def gauss_predict(Z, te, sid, Ms, Prec, per_subject_prec=False):
    out = np.empty(len(te), int)
    for s in np.unique(sid[te]):
        pos = np.where(sid[te] == s)[0]
        P = Prec[s] if per_subject_prec else Prec
        A = Ms[s] @ P
        out[pos] = (Z[te[pos]] @ A.T - 0.5 * np.sum(A * Ms[s], 1)).argmax(1)
    return out


def pick_gamma(Z, y, sid, tr, n_cls, n_sub, grid, inner=3, seed=0):
    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid[tr], y[tr])])
    score = np.zeros(len(grid)); folds = 0
    for itr, ite in StratifiedKFold(inner, shuffle=True, random_state=seed).split(tr, strat):
        a, b = tr[itr], tr[ite]
        M, N, Mp = cell_means(Z, y, sid, a, n_cls, n_sub)
        P = shared_precision(Z, y, sid, a, M)
        for j, g in enumerate(grid):
            score[j] += balanced_accuracy_score(
                y[b], gauss_predict(Z, b, sid, blend(M, Mp, g), P))
        folds += 1
    return float(grid[int(np.argmax(score))]), score / max(folds, 1)


# ---------------------------------------------------------------- two-parameter family

LAM = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])     # noise sharing
GAM = np.linspace(0, 1, 11)                                # class sharing
KAP = np.array([0.01, 0.05, 0.15, 0.4])                    # ridge


def _sym_inv_sqrt(S, floor=1e-10):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, max(w.max(), 1e-12) * floor, None)
    return (V / np.sqrt(w)) @ V.T


def prepare(Z, y, sid, tr, n_cls, n_sub):
    """Per-subject eigen-bases covering the whole (lambda, kappa) grid at once.

    Sigma_s(l) = (1-l)Sigma_s + l Sigma_group = W^-1[(1-l) W Sigma_s W + l I] W^-1 with
    W = Sigma_group^-1/2, and every member of that family shares the eigenvectors of
    W Sigma_s W. One eigendecomposition per subject therefore covers the grid and scoring
    reduces to a diagonal reweighting.
    """
    M, N, Mp = cell_means(Z, y, sid, tr, n_cls, n_sub)
    Rg = Z[tr] - M[sid[tr], y[tr]]
    Sg, _ = ledoit_wolf(Rg, assume_centered=True)
    W = _sym_inv_sqrt(Sg)
    basis = {}
    for s in range(n_sub):
        i = tr[sid[tr] == s]
        if len(i) < 2 * n_cls:
            continue
        R = Z[i] - M[sid[i], y[i]]
        Ss = R.T @ R / max(len(i) - 1, 1)
        a, U = np.linalg.eigh(W @ Ss @ W)
        basis[s] = (np.clip(a, 0.0, None), W @ U)
    return M, Mp, basis


def band_scores(Z, sid, idx, M, Mp, basis, n_cls, l, g, k):
    """Gaussian log-scores up to a class-independent constant, for one feature block."""
    S = np.zeros((len(idx), n_cls))
    for s in np.unique(sid[idx]):
        if s not in basis:
            continue
        a, G = basis[s]
        pos = np.where(sid[idx] == s)[0]
        Zt = Z[idx[pos]] @ G
        Mb = g * (Mp @ G) + (1 - g) * (M[s] @ G)
        e0 = (1 - l) * a + l
        e = np.clip((1 - k) * e0 + k * e0.mean(), 1e-12, None)
        A = Mb / e
        S[pos] = Zt @ A.T - 0.5 * np.sum(A * Mb, 1)
    return S


def score_grid(Z, y, sid, idx, M, Mp, basis, n_cls, lam=LAM, gam=GAM, kap=KAP):
    """Balanced accuracy for every (lambda, gamma, kappa)."""
    hit = np.zeros((len(lam), len(gam), len(kap), n_cls))
    tot = np.array([np.sum(y[idx] == c) for c in range(n_cls)], float)
    for s in np.unique(sid[idx]):
        if s not in basis:
            continue
        a, G = basis[s]
        i = idx[sid[idx] == s]
        Zt = Z[i] @ G; Mt = M[s] @ G; Pt = Mp @ G; yt = y[i]
        for li, l in enumerate(lam):
            e0 = (1 - l) * a + l
            for ki, k in enumerate(kap):
                e = np.clip((1 - k) * e0 + k * e0.mean(), 1e-12, None)
                for gi, g in enumerate(gam):
                    Mb = g * Pt + (1 - g) * Mt
                    A = Mb / e
                    pred = (Zt @ A.T - 0.5 * np.sum(A * Mb, 1)).argmax(1)
                    for c in range(n_cls):
                        hit[li, gi, ki, c] += np.sum((pred == c) & (yt == c))
    ok = tot > 0
    return (hit[..., ok] / tot[ok]).mean(-1)


def tune_head(Z, y, sid, tr, n_cls, n_sub, inner=3, seed=0):
    """(lambda, gamma, kappa) by inner CV on the training block only."""
    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid[tr], y[tr])])
    surf = np.zeros((len(LAM), len(GAM), len(KAP)))
    for itr, ite in StratifiedKFold(inner, shuffle=True, random_state=seed).split(tr, strat):
        M, Mp, basis = prepare(Z, y, sid, tr[itr], n_cls, n_sub)
        surf += score_grid(Z, y, sid, tr[ite], M, Mp, basis, n_cls)
    surf /= inner
    li, gi, ki = np.unravel_index(surf.argmax(), surf.shape)
    return float(LAM[li]), float(GAM[gi]), float(KAP[ki]), float(surf.max()), surf


def head(F, y, sid, tr, te, n_cls, n_sub, inner=3, seed=0):
    """Tune on the training folds, refit on all of them, score the test fold."""
    l, g, kk, _, _ = tune_head(F, y, sid, tr, n_cls, n_sub, inner, seed)
    M, Mp, basis = prepare(F, y, sid, tr, n_cls, n_sub)
    acc = balanced_accuracy_score(
        y[te], band_scores(F, sid, te, M, Mp, basis, n_cls, l, g, kk).argmax(1))
    return acc, (l, g, kk)


# ---------------------------------------------------------------- reporting

def report(res, chance, ref=None, label="", width=22):
    """One table: mean, vs-chance, and paired against `ref`. `res` maps name -> per-fold list.

    Sorted by mean because the ordering is the result, and the paired p is against a named
    reference so that "beats" always means "beats a specific stated alternative".
    """
    names = [k for k in res if not np.all(np.isnan(np.asarray(res[k], float)))]
    if not names:
        return
    if ref is None or ref not in res:
        ref = max(names, key=lambda k: np.nanmean(res[k]))
    r = np.asarray(res[ref], float)
    if label:
        print("  --- %s (reference: %s) ---" % (label, ref))
    for k in sorted(names, key=lambda k: -np.nanmean(res[k])):
        v = np.asarray(res[k], float)
        _, p0 = ttest_1samp(v[~np.isnan(v)], chance)
        if len(r) == len(v):
            _, pb = ttest_rel(v, r)
            extra = "  vs-%s %+.3f (p=%.4f) %s" % (
                ref[:12], np.nanmean(v) - np.nanmean(r), pb,
                "**BEATS**" if pb < 0.05 and np.nanmean(v) > np.nanmean(r) else "")
        else:
            extra = ""
        print("  %-*s %.3f  (%d folds)  vs-chance p=%.7f%s" % (
            width, k, np.nanmean(v), np.sum(~np.isnan(v)), p0, extra))
