import numpy as np


def _legendre(x: np.ndarray, nmax: int) -> np.ndarray:
    P = np.zeros((nmax + 1,) + x.shape)
    P[0] = 1.0
    if nmax >= 1:
        P[1] = x
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return P


def _gh(cosang: np.ndarray, m: int = 4, nmax: int = 50) -> tuple[np.ndarray, np.ndarray]:
    P = _legendre(np.clip(cosang, -1.0, 1.0), nmax)
    n = np.arange(1, nmax + 1)
    wg = (2 * n + 1) / (n ** m * (n + 1) ** m)
    wh = (2 * n + 1) / (n ** (m - 1) * (n + 1) ** (m - 1))
    g = np.tensordot(wg, P[1:], axes=(0, 0)) / (4 * np.pi)
    h = -np.tensordot(wh, P[1:], axes=(0, 0)) / (4 * np.pi)
    return g, h


def csd_operator(pos: np.ndarray, m: int = 4, lam: float = 1e-5,
                 nmax: int = 50) -> np.ndarray:
    U = np.asarray(pos, float)
    U = U - U.mean(0)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    G, H = _gh(U @ U.T, m, nmax)
    n = len(U)
    Gs = G + lam * np.eye(n)
    Gi = np.linalg.inv(Gs)
    one = np.ones((n, 1))
    d = float(one.T @ Gi @ one)
    T = Gi - (Gi @ one @ one.T @ Gi) / d
    return H @ T


def apply_csd(X: np.ndarray, L: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njt->nit", L, X)


if __name__ == "__main__":
    import glob
    f = sorted(glob.glob("npz2/*.npz"))[0]
    d = np.load(f, allow_pickle=True)
    pos = np.asarray(d["channel_positions_xyz"], float)
    L = csd_operator(pos)
    n = len(pos)
    print("channels %d   operator %s" % (n, L.shape))
    print("row sums (should be ~0):  max |sum| = %.3e" % np.abs(L.sum(1)).max())
    V = np.ones((1, n, 5))
    print("CSD of uniform field:     max |val| = %.3e" % np.abs(apply_csd(V, L)).max())
    rng = np.random.default_rng(0)
    a = rng.normal(size=3)
    grad = (pos - pos.mean(0)) @ a
    print("CSD of linear gradient:   rel mag  = %.3e"
          % (np.abs(L @ grad).max() / np.abs(grad).max()))
    X = d["X"].astype(float)[:50]
    Y = apply_csd(X, L)
    c = np.corrcoef(np.abs(np.corrcoef(X.transpose(1, 0, 2).reshape(n, -1))[np.triu_indices(n, 1)]),
                    np.abs(np.corrcoef(Y.transpose(1, 0, 2).reshape(n, -1))[np.triu_indices(n, 1)]))[0, 1]
    print("mean |inter-channel r|:   raw %.3f -> csd %.3f   (corr of patterns %.3f)"
          % (np.abs(np.corrcoef(X.transpose(1, 0, 2).reshape(n, -1))[np.triu_indices(n, 1)]).mean(),
             np.abs(np.corrcoef(Y.transpose(1, 0, 2).reshape(n, -1))[np.triu_indices(n, 1)]).mean(), c))
