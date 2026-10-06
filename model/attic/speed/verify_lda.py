import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
import fastlda as fl

rng = np.random.default_rng(0)
n, C = 176, 11
y = np.repeat(np.arange(C), n // C)
yt = np.repeat(np.arange(C), 4)

print("=== 1. Ledoit-Wolf shrinkage: mine vs sklearn (class-wise) ===")
for p in [300, 759, 1518]:
    Z = rng.normal(size=(16, p))
    mine, _ = fl._lw_shrinkage(Z - Z.mean(0))
    ref = ledoit_wolf_shrinkage(Z, assume_centered=False)
    print("  p=%-5d n=16   mine=%.6f   sklearn=%.6f   diff=%.2e" % (p, mine, ref, abs(mine - ref)))

print("\n=== 2. weak-signal regime (our actual accuracy range) ===")
for p, s in [(300, 0.05), (759, 0.04), (1518, 0.03)]:
    sig = rng.normal(size=(C, p)) * s
    X = sig[y] + rng.normal(size=(n, p))
    Xt = sig[yt] + rng.normal(size=(len(yt), p))
    t0 = time.time()
    sk = LDA(solver="lsqr", shrinkage="auto").fit(X, y)
    ps = sk.predict(Xt)
    t1 = time.time()
    co, it, cl = fl.fit_dual(X, y)
    pd = fl.predict(Xt, co, it, cl)
    t2 = time.time()
    ds = sk.decision_function(Xt)
    dd = Xt @ co.T + it
    rho = np.corrcoef(ds.ravel(), dd.ravel())[0, 1]
    print("  p=%-5d acc sk=%.4f dual=%.4f   agree=%.4f   decision_corr=%.8f   %.2fs->%.3fs (%.0fx)"
          % (p, balanced_accuracy_score(yt, ps), balanced_accuracy_score(yt, pd),
             (ps == pd).mean(), rho, t1 - t0, t2 - t1, (t1 - t0) / max(t2 - t1, 1e-9)))

print("\n=== 3. dual-only timing at target dims ===")
for p in [2970, 5940]:
    X = rng.normal(size=(n, p))
    t0 = time.time()
    fl.fit_dual(X, y)
    print("  p=%-5d  %.3fs   (sklearn extrapolated %.0fs)" % (p, time.time() - t0,
                                                             14.9 * (p / 1518) ** 3))
