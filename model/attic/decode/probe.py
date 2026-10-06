import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import unified as U
import fastlda as fl
import csd as csdmod
from sklearn.metrics import balanced_accuracy_score

TARGET = ["S06", "S07", "S11", "S12"]
NPERM = 5


def run(path, tn, perm_seed=None, nseed=1):
    X, ph, pos = U.load(path, 0.2, 1.01)
    L = csdmod.csd_operator(pos)
    keep = U.STOPS if tn == "place" else U.PHON
    m = np.isin(ph, keep)
    p = ph[m]
    y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
         else np.array([U.PHON.index(v) for v in p]))
    if perm_seed is not None:
        y = np.random.default_rng(perm_seed).permutation(y)
    XR = {False: X[m], True: csdmod.apply_csd(X[m], L)}
    acc = {g: [] for g in U.GRID}
    M0 = {g: None for g in U.GRID}
    for sd in range(nseed):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(XR[False], y):
            for c in U.CSDS:
                for nf in U.NFS:
                    Za, Zb = U.augment(XR[c][tr], y[tr], XR[c][te], nf)
                    nch, T = Za.shape[1], Za.shape[2]
                    for K in U.KS:
                        if nch / (T // K) > U.GMAX:
                            continue
                        g = (c, nf, K)
                        try:
                            a, b, M0[g] = U.feats(Za, Zb, K, M0[g])
                            sc = StandardScaler().fit(a)
                            co, it, cl = fl.fit(sc.transform(a), y[tr])
                            acc[g].append(balanced_accuracy_score(
                                y[te], fl.predict(sc.transform(b), co, it, cl)))
                        except Exception:
                            M0[g] = None
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in acc.items()}


def job(a):
    path, tn, ps = a
    return os.path.basename(path)[:3], tn, ps, run(path, tn, ps)


if __name__ == "__main__":
    F = [f for f in sorted(glob.glob("npz2/*.npz")) if os.path.basename(f)[:3] in TARGET]
    jobs = [(f, "ph11", None) for f in F]
    s11 = [f for f in F if "S11" in f][0]
    jobs += [(s11, "ph11", 100 + i) for i in range(NPERM)]
    print("PROBE | real: %s | permutation control on S11 x%d\n" % (TARGET, NPERM), flush=True)
    real, perm = {}, []
    with Pool(min(9, len(jobs))) as pool:
        for sub, tn, ps, o in pool.imap_unordered(job, jobs):
            if ps is None:
                real[sub] = o
                print("  REAL %s  " % sub + "  ".join(
                    "%s%d%d=%.3f" % ("C" if g[0] else "r", g[1], g[2], o[g])
                    for g in sorted(o) if not np.isnan(o[g])), flush=True)
            else:
                perm.append(o)
                v = [x for x in o.values() if not np.isnan(x)]
                print("  PERM S11 seed=%d  max=%.3f  mean=%.3f" % (ps, max(v), np.mean(v)),
                      flush=True)
    print("\n=== permutation control (S11, labels shuffled, chance=0.0909) ===")
    mx = [max(x for x in o.values() if not np.isnan(x)) for o in perm]
    mn = [np.nanmean(list(o.values())) for o in perm]
    print("  max-over-configs : mean %.4f   range %.3f-%.3f" % (np.mean(mx), min(mx), max(mx)))
    print("  mean-over-configs: mean %.4f" % np.mean(mn))
    print("\n=== real S11 vs its own permutation ceiling ===")
    if "S11" in real:
        v = {g: x for g, x in real["S11"].items() if not np.isnan(x)}
        print("  real best %.4f (%s)   perm best %.4f   gap %+.4f"
              % (max(v.values()), max(v, key=v.get), np.mean(mx),
                 max(v.values()) - np.mean(mx)))
