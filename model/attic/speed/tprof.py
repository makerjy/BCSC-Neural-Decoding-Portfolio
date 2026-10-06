import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import fastriem as fr

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
NPCA = {"place": 20, "ph11": 15}
CHANCE = {"place": 0.5, "ph11": 1 / 11}
WLEN, STEP = 0.2, 0.05
STARTS = np.round(np.arange(0.0, 0.801, STEP), 3)


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    X = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    return X, t, d["phoneme1"].astype(str)[k]


def ev(Xs, y, npca):
    acc, M0 = [], None
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        pc = PCA(min(npca, len(tr) - 2)).fit(
            Xs[tr].transpose(0, 2, 1).reshape(-1, Xs.shape[1]))
        pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
        Ca, Cb = fr.cov_oas(pr(Xs[tr])), fr.cov_oas(pr(Xs[te]))
        M0 = fr.riemann_mean(Ca, init=M0)
        a, b = fr.tangent(Ca, M0), fr.tangent(Cb, M0)
        mm = np.concatenate([pr(Xs[tr]).mean(2), np.log(pr(Xs[tr]).var(2) + 1e-12)], 1)
        mn = np.concatenate([pr(Xs[te]).mean(2), np.log(pr(Xs[te]).var(2) + 1e-12)], 1)
        m1 = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(a, y[tr])
        m2 = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(mm, y[tr])
        acc.append((balanced_accuracy_score(y[te], m1.predict(b)),
                    balanced_accuracy_score(y[te], m2.predict(mn))))
    return tuple(np.mean(acc, 0))


def one(f):
    X, t, ph = load(f)
    out = {}
    for tn, keep in [("place", STOPS), ("ph11", PHON)]:
        m = np.isin(ph, keep)
        p = ph[m]
        y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
             else np.array([PHON.index(v) for v in p]))
        for s in STARTS:
            msk = (t >= s) & (t < s + WLEN)
            out[(tn, float(s))] = ev(X[m][:, :, msk], y, NPCA[tn])
    return os.path.basename(f)[:3], out


if __name__ == "__main__":
    FILES = sorted(glob.glob("npz2/*.npz"))
    print("TIME PROFILE | %d subj | %.0f ms window, %.0f ms step\n"
          % (len(FILES), WLEN * 1000, STEP * 1000), flush=True)
    res = []
    with Pool(16) as pool:
        for sub, o in pool.imap_unordered(one, FILES):
            res.append((sub, o))
            best = max(STARTS, key=lambda s: o[("ph11", float(s))][0])
            print("  %s done  ph11 peak @ %d-%d ms = %.3f"
                  % (sub, best * 1000, (best + WLEN) * 1000, o[("ph11", float(best))][0]),
                  flush=True)
    for tn in ["place", "ph11"]:
        print("\n=== %s (chance %.3f) ===" % (tn, CHANCE[tn]))
        print("  window(ms)      cov    mean+logvar   p_vs_chance(cov)")
        for s in STARTS:
            v = np.array([o[(tn, float(s))] for _, o in res])
            p = ttest_1samp(v[:, 0] - CHANCE[tn], 0).pvalue
            bar = "#" * int(max(0, (v[:, 0].mean() - CHANCE[tn]) * (200 if tn == "ph11" else 100)))
            print("  %4d-%4d   %.3f     %.3f       %.2e  %s"
                  % (s * 1000, (s + WLEN) * 1000, v[:, 0].mean(), v[:, 1].mean(), p, bar))
