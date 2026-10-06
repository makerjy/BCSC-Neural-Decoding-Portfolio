import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from scipy.stats import spearmanr

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
rng = np.random.default_rng(0)


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    Xk = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    y = d["phoneme1"].astype(str)[k]
    g = np.isin(y, PHON)
    pos = d["channel_positions_xyz"] if "channel_positions_xyz" in d else None
    return Xk[g], y[g], X, pos


def rank_of(X, tol=1e-6):
    Z = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
    ev = np.linalg.eigvalsh(np.cov(Z - Z.mean(1, keepdims=True)))[::-1]
    return int((ev / ev[0] > tol).sum())


def reliab(X, y, nrep=30):
    ev, dv = [], []
    for _ in range(nrep):
        h = [[], []]
        for c in PHON:
            i = rng.permutation(np.where(y == c)[0])
            h[0].append(i[:len(i) // 2])
            h[1].append(i[len(i) // 2:])
        M = [np.stack([X[j].mean(0) for j in hh], 0) for hh in h]
        ev.append(np.mean([np.corrcoef(M[0][k].ravel(), M[1][k].ravel())[0, 1]
                           for k in range(11)]))
        D = [m - m.mean(0, keepdims=True) for m in M]
        dv.append(np.mean([np.corrcoef(D[0][k].ravel(), D[1][k].ravel())[0, 1]
                           for k in range(11)]))
    return float(np.mean(ev)), float(np.mean(dv))


F = sorted(glob.glob("npz2/*.npz"))
print("ACCURACY-INDEPENDENT DATA QUALITY (no classifier anywhere)\n")
print("  subj  ntr  rank  ICAdrop  evoked_rel  class_dev_rel  amp_p99(uV)  badtrial%")
rows = []
P0 = None
for f in F:
    X, y, Xall, pos = load(f)
    r = rank_of(Xall)
    e, c = reliab(X, y)
    amp = np.percentile(np.abs(X), 99)
    tmax = np.abs(X).max((1, 2))
    bad = float(np.mean(tmax > 5 * np.median(tmax))) * 100
    rows.append((os.path.basename(f)[:3], len(y), r, 61 - r, e, c, amp, bad))
    if pos is not None:
        P0 = pos if P0 is None else P0
        same = np.allclose(np.asarray(pos, float), np.asarray(P0, float))
    print("  %-4s %4d  %3d   %3d      %+.3f      %+.3f       %7.1f    %5.1f"
          % rows[-1])

a = np.array([r[1:] for r in rows], float)
nm = ["ntr", "rank", "ICAdrop", "evoked_rel", "class_dev_rel", "amp_p99", "badtrial%"]
print("\n  median:", "  ".join("%s=%.3f" % (n, v) for n, v in zip(nm, np.median(a, 0))))
print("\n  positions identical across subjects:", bool(same))
print("\n=== flagged by class_dev_rel < 0.02 (pre-set, accuracy-independent) ===")
flag = [r[0] for r in rows if r[5] < 0.02]
print("  ", flag if flag else "none")
print("=== flagged by evoked_rel < 0.10 ===")
print("  ", [r[0] for r in rows if r[4] < 0.10])
print("\n  rank(class_dev_rel) vs rank(ICAdrop)  rho=%+.3f"
      % spearmanr(a[:, 4], a[:, 2]).statistic)
