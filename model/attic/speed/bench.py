import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import fastriem as fr

rng = np.random.default_rng(0)
n, C = 176, 11
y = np.repeat(np.arange(C), n // C)
print("n=%d  classes=%d   (per class %d)\n" % (len(y), C, n // C))
print("  p        LDA fit+predict     per-fold x15")
for p in [253 * 3, 253 * 6, 990 * 3, 990 * 6]:
    X = rng.normal(size=(len(y), p))
    Xt = rng.normal(size=(44, p))
    t0 = time.time()
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(X, y)
    m.predict(Xt)
    dt = time.time() - t0
    print("  %-7d  %6.2f s            %6.1f s" % (p, dt, dt * 15))

print("\n  batched eigh (riemann_mean / tangent 내부)")
for d, N in [(22, 176), (44, 176)]:
    A = rng.normal(size=(N, d, 40))
    t0 = time.time()
    Cv = fr.cov_oas(A)
    M = fr.riemann_mean(Cv)
    V = fr.tangent(Cv, M)
    dt = time.time() - t0
    print("    d=%2d N=%d  cov+mean+tangent %6.3f s   x6 windows x15 folds = %5.1f s"
          % (d, N, dt, dt * 90))
