import numpy as np


def cov_oas(X: np.ndarray) -> np.ndarray:
    n, d, T = X.shape
    Xc = X - X.mean(2, keepdims=True)
    S = Xc @ Xc.transpose(0, 2, 1) / T
    tr = np.trace(S, axis1=1, axis2=2)
    mu = tr / d
    alpha = (S ** 2).sum((1, 2)) / (d * d)
    num = alpha + mu ** 2
    den = (T + 1.0) * (alpha - (mu ** 2) / d)
    rho = np.where(den > 0, np.minimum(num / den, 1.0), 1.0)
    out = (1.0 - rho)[:, None, None] * S
    idx = np.arange(d)
    out[:, idx, idx] += (rho * mu)[:, None]
    return out


def _eig(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(C)
    return np.maximum(w, 1e-14), V


def powm(C: np.ndarray, p: float) -> np.ndarray:
    w, V = _eig(C)
    return (V * (w ** p)[..., None, :]) @ V.swapaxes(-1, -2)


def logm(C: np.ndarray) -> np.ndarray:
    w, V = _eig(C)
    return (V * np.log(w)[..., None, :]) @ V.swapaxes(-1, -2)


def expm(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * np.exp(w)[..., None, :]) @ V.swapaxes(-1, -2)


def riemann_mean(C: np.ndarray, tol: float = 1e-8, maxit: int = 60,
                 init: np.ndarray | None = None) -> np.ndarray:
    M = C.mean(0) if init is None else init.copy()
    nu, tau, crit = 1.0, np.finfo(np.float64).max, np.finfo(np.float64).max
    for _ in range(maxit):
        Mis = powm(M, -0.5)
        Ms = powm(M, 0.5)
        J = logm(Mis @ C @ Mis).mean(0)
        crit_new = np.linalg.norm(J, ord="fro")
        h = nu * crit_new
        M = Ms @ expm(nu * J) @ Ms
        M = (M + M.T) / 2
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit_new <= tol or nu <= tol:
            break
        crit = crit_new
    return M


def tangent(C: np.ndarray, M: np.ndarray) -> np.ndarray:
    d = C.shape[-1]
    Mis = powm(M, -0.5)
    L = logm(Mis @ C @ Mis)
    iu = np.triu_indices(d, 1)
    di = np.arange(d)
    return np.concatenate([L[:, di, di], np.sqrt(2.0) * L[:, iu[0], iu[1]]], 1)
