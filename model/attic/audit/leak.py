import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_rel, t as tdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import fastriem as fr

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
NPCA, K = 25, 3
WINS = {"early": (0.0, 0.5), "late": (0.5, 1.1)}


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    tn = d["trial_n"].astype(int)
    o = np.argsort(tn)
    X, tn = X[o], tn[o]
    tk, tg = d["task"].astype(str)[o], d["tms_target"].astype(str)[o]
    p1 = d["phoneme1"].astype(str)[o]
    X = X - X[:, :, t <= 0.02].mean(2, keepdims=True)
    k = (tk == "single_phoneme") & (tg == "control") & np.isin(p1, PHON)
    return X, t, tn, k, p1


def feat(Xtr, Xte, M0):
    pc = PCA(min(NPCA, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    va, vb, Ms = [], [], []
    for i in range(K):
        Ca = fr.cov_oas(A[:, :, ed[i]:ed[i + 1]])
        Cb = fr.cov_oas(B[:, :, ed[i]:ed[i + 1]])
        M = fr.riemann_mean(Ca, init=None if M0 is None else M0[i])
        Ms.append(M)
        va.append(fr.tangent(Ca, M))
        vb.append(fr.tangent(Cb, M))
    return np.concatenate(va, 1), np.concatenate(vb, 1), Ms


def ev(Xs, y):
    acc, M0 = [], None
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        a, b, M0 = feat(Xs[tr], Xs[te], M0)
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(a, y[tr])
        acc.append(balanced_accuracy_score(y[te], m.predict(b)))
    return float(np.mean(acc))


def one(f):
    X, t, tn, k, p1 = load(f)
    idx = np.where(k)[0]
    pair = [(i, j) for i, j in zip(idx[:-1], idx[1:]) if tn[j] - tn[i] == 1]
    cur = np.array([a for a, _ in pair])
    nxt = np.array([b for _, b in pair])
    yo = np.array([PHON.index(v) for v in p1[cur]])
    yn = np.array([PHON.index(v) for v in p1[nxt]])
    rep = float(np.mean(yo == yn))
    out = {"n_pair": len(pair), "repeat_rate": rep}
    for wn, (t0, t1) in WINS.items():
        Xs = X[cur][:, :, (t >= t0) & (t < t1)]
        out[(wn, "own")] = ev(Xs, yo)
        out[(wn, "next")] = ev(Xs, yn)
    return os.path.basename(f)[:3], out


if __name__ == "__main__":
    FILES = sorted(glob.glob("npz2/*.npz"))
    print("LEAKAGE TEST: does the late window contain the NEXT stimulus?\n", flush=True)
    res = []
    with Pool(16) as pool:
        for sub, o in pool.imap_unordered(one, FILES):
            res.append((sub, o))
            print("  %s n=%d rep=%.3f | early own=%.3f next=%.3f | late own=%.3f next=%.3f"
                  % (sub, o["n_pair"], o["repeat_rate"], o[("early", "own")],
                     o[("early", "next")], o[("late", "own")], o[("late", "next")]), flush=True)

    def ci(d):
        m, s, n = d.mean(), d.std(ddof=1), len(d)
        h = tdist.ppf(.975, n - 1) * s / np.sqrt(n)
        return m, m - h, m + h

    G = lambda k: np.array([o[k] for _, o in res])
    print("\n=== chance 0.0909 | immediate-repeat rate %.4f (uniform would be 0.0909) ==="
          % np.mean([o["repeat_rate"] for _, o in res]))
    for k in [("early", "own"), ("late", "own"), ("early", "next"), ("late", "next")]:
        v = G(k)
        m, lo, hi = ci(v - 1 / 11)
        print("  %-12s %.3f   vs chance %+.3f [%+.3f,%+.3f]" % ("%s->%s" % k, v.mean(), m, lo, hi))
    d = G(("late", "next")) - G(("early", "next"))
    m, lo, hi = ci(d)
    print("\n  CONTAMINATION  late->next minus early->next  %+.4f  95%%CI [%+.4f,%+.4f]  p=%.4f"
          % (m, lo, hi, ttest_rel(G(("late", "next")), G(("early", "next"))).pvalue))
