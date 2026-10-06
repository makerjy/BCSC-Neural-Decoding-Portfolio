import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

SUBS = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
VOWELS = ["a", "e", "i", "o", "u"]
STOPS = ["b", "p", "d", "t"]

def load(sub):
    d = np.load("npz/sub-%s_ses-02.npz" % sub, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def ev_syl(sub):
    ev = pd.read_csv("events/sub-%s_phonemes_events.tsv" % sub, sep="\t")
    st = ev[ev.trial_type == "stimulus"].reset_index(drop=True)
    return st.phoneme1.astype(str).values, st.phoneme2.astype(str).str.strip().values

def cut(X, t, mask, t0, t1, do_base=True):
    wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    if do_base:
        bm = (t >= -0.2) & (t <= -0.11)
        Xm = Xm - Xm[:, :, bm].mean(2, keepdims=True)
    else:
        Xm = Xm - Xm.mean(2, keepdims=True)
    return Xm[:, :, wm]

def feats(Xtr, Xte, npca=20):
    pc = PCA(min(npca, len(Xtr)-2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = prj(Xtr), prj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, 3).astype(int)
    fa, fb = [], []
    for i in range(2):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def run(Xtr, ytr, Xte, yte):
    a, b = feats(Xtr, Xte)
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(a, ytr); return balanced_accuracy_score(yte, c.predict(b))

def rand_cv(Xw, y, seeds=4, folds=3):
    if min(np.bincount(y)) < 4: return np.nan
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=s).split(Xw, y):
            acc.append(run(Xw[tr], y[tr], Xw[te], y[te]))
    return float(np.mean(acc))

def block_cv(Xw, y, nb=3):
    n = len(y); acc = []
    ed = np.linspace(0, n, nb+1).astype(int)
    for i in range(nb):
        te = np.arange(ed[i], ed[i+1]); tr = np.setdiff1d(np.arange(n), te)
        if len(np.unique(y[te])) < 2 or min(np.bincount(y[tr])) < 4: continue
        acc.append(run(Xw[tr], y[tr], Xw[te], y[te]))
    return float(np.mean(acc)) if acc else np.nan

R = {k: [] for k in ["v_rand", "v_block", "v_pre", "order", "runs_v", "runs_c"]}
print("CONFOUND AUDIT on SYLLABLE (phonemes CV) data\n")
for sub in SUBS:
    X, t, meta = load(sub)
    task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_all = task == "phonemes"
    try:
        p1, p2 = ev_syl(sub)
    except Exception:
        continue
    if m_all.sum() != len(p1): continue
    sel = ctrl[m_all] & np.isin(p2, VOWELS)
    Xs = cut(X, t, m_all, 0.0, 0.5)[sel]
    y = np.array([VOWELS.index(v) for v in p2[sel]])
    R["v_rand"].append(rand_cv(Xs, y))
    R["v_block"].append(block_cv(Xs, y))
    Xpre = cut(X, t, m_all, -0.2, -0.05, do_base=False)[sel]
    R["v_pre"].append(rand_cv(Xpre, y, seeds=2))
    half = (np.arange(sel.sum()) > sel.sum()//2).astype(int)
    R["order"].append(rand_cv(Xs, half, seeds=2))
    vv = p2[sel]; cc = p1[sel]
    R["runs_v"].append(np.mean(vv[1:] == vv[:-1]))
    R["runs_c"].append(np.mean(cc[1:] == cc[:-1]))
    print("  %-5s vowel rand %.3f | chrono %.3f | PRE %.3f | order %.3f | repeat-rate V %.2f C %.2f" % (
        sub, R["v_rand"][-1], R["v_block"][-1], R["v_pre"][-1], R["order"][-1],
        R["runs_v"][-1], R["runs_c"][-1]), flush=True)

print("\n=== SYLLABLE AUDIT (vowel 5-way, chance 0.2) ===")
for k, ch, lbl in [("v_rand", .2, "vowel, random CV"), ("v_block", .2, "vowel, CHRONOLOGICAL CV"),
                   ("v_pre", .2, "vowel from PRE-STIMULUS"), ("order", .5, "trial order early-vs-late")]:
    a = np.array([v for v in R[k] if not np.isnan(v)])
    _, p = ttest_1samp(a, ch)
    print("  %-34s %.3f +- %.3f  (%d/%d)  p=%.5f %s" % (
        lbl, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))
print("\n  consecutive-repeat rate: vowel %.3f, consonant %.3f  (0.2/0.25 = random order)" % (
    np.mean(R["runs_v"]), np.mean(R["runs_c"])))
