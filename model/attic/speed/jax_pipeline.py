from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)


def spd_sqrt(C: ArrayLike) -> jax.Array:
    w, V = jnp.linalg.eigh(C)
    w = jnp.clip(w, 1e-10, None)
    return (V * jnp.sqrt(w)[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def spd_invsqrt(C: ArrayLike) -> jax.Array:
    w, V = jnp.linalg.eigh(C)
    w = jnp.clip(w, 1e-10, None)
    return (V * (1.0 / jnp.sqrt(w))[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def spd_log(C: ArrayLike) -> jax.Array:
    w, V = jnp.linalg.eigh(C)
    w = jnp.clip(w, 1e-10, None)
    return (V * jnp.log(w)[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def spd_exp(S: ArrayLike) -> jax.Array:
    w, V = jnp.linalg.eigh(S)
    return (V * jnp.exp(w)[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def covariances(X: ArrayLike, shrink: float | None = None) -> jax.Array:
    n_samp = X.shape[-1]
    d = X.shape[-2]
    Xc = X - X.mean(axis=-1, keepdims=True)
    C = jnp.einsum("...ct,...dt->...cd", Xc, Xc) / n_samp
    mu = jnp.trace(C, axis1=-2, axis2=-1)[..., None, None] / d
    if shrink is not None:
        return (1.0 - shrink) * C + shrink * mu * jnp.eye(d)
    alpha = jnp.mean(C * C, axis=(-2, -1))
    mu_s = jnp.squeeze(mu, (-2, -1))
    num = alpha + mu_s ** 2
    den = (n_samp + 1.0) * (alpha - (mu_s ** 2) / d)
    rho = jnp.where(den > 0, jnp.clip(num / jnp.where(den > 0, den, 1.0), 0.0, 1.0), 1.0)
    rho = rho[..., None, None]
    return (1.0 - rho) * C + rho * mu * jnp.eye(d)


def karcher_mean(Cs: ArrayLike, n_iter: int = 12) -> jax.Array:
    M0 = jnp.mean(Cs, axis=0)

    def step(M: jax.Array, _: None) -> tuple[jax.Array, None]:
        Mh, Mih = spd_sqrt(M), spd_invsqrt(M)
        S = jnp.mean(spd_log(jnp.einsum("ij,njk,kl->nil", Mih, Cs, Mih)), axis=0)
        return Mh @ spd_exp(S) @ Mh, None

    M, _ = lax.scan(step, M0, None, length=n_iter)
    return M


def tangent_space(Cs: ArrayLike, ref: ArrayLike) -> jax.Array:
    Rih = spd_invsqrt(ref)
    L = spd_log(jnp.einsum("ij,njk,kl->nil", Rih, Cs, Rih))
    d = L.shape[-1]
    iu = jnp.triu_indices(d)
    scale = jnp.where(iu[0] == iu[1], 1.0, jnp.sqrt(2.0))
    return L[:, iu[0], iu[1]] * scale


def pca_fit(X: ArrayLike, n_comp: int) -> jax.Array:
    flat = jnp.transpose(X, (0, 2, 1)).reshape(-1, X.shape[1])
    flat = flat - flat.mean(0)
    _, _, Vt = jnp.linalg.svd(flat, full_matrices=False)
    return Vt[:n_comp]


def pca_apply(X: ArrayLike, comps: ArrayLike) -> jax.Array:
    return jnp.einsum("kc,nct->nkt", comps, X)


def windowed_riemann(Xtr: ArrayLike, Xte: ArrayLike, n_win: int, n_pca: int) -> tuple[jax.Array, jax.Array]:
    comps = pca_fit(Xtr, n_pca)
    A, B = pca_apply(Xtr, comps), pca_apply(Xte, comps)
    T = A.shape[2]
    bounds = jnp.linspace(0, T, n_win + 1).astype(int)
    fa, fb = [], []
    for i in range(n_win):
        s, e = int(bounds[i]), int(bounds[i + 1])
        Ca, Cb = covariances(A[:, :, s:e]), covariances(B[:, :, s:e])
        ref = karcher_mean(Ca)
        fa.append(tangent_space(Ca, ref))
        fb.append(tangent_space(Cb, ref))
    return jnp.hstack(fa), jnp.hstack(fb)


def reservoir_run(U: ArrayLike, W: ArrayLike, Win: ArrayLike, leaks: ArrayLike,
                  sub: int = 20) -> jax.Array:
    n = U.shape[0]
    h0 = jnp.zeros((n, W.shape[0]))

    def step(h: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        h_new = (1.0 - leaks) * h + leaks * jnp.tanh(h @ W.T + u @ Win.T)
        return h_new, h_new

    _, H = lax.scan(step, h0, jnp.transpose(U, (1, 0, 2)))
    return jnp.transpose(H[::sub], (1, 0, 2)).reshape(n, -1)


def shrinkage_lda_fit(F: ArrayLike, y: ArrayLike, n_class: int,
                      shrink: float = 0.2) -> tuple[jax.Array, jax.Array]:
    d = F.shape[1]
    mu = jnp.stack([jnp.where((y == k)[:, None], F, 0.0).sum(0) /
                    jnp.maximum((y == k).sum(), 1) for k in range(n_class)])
    Xc = F - mu[y]
    S = (Xc.T @ Xc) / jnp.maximum(F.shape[0] - n_class, 1)
    S = (1.0 - shrink) * S + shrink * jnp.trace(S) / d * jnp.eye(d)
    Si = jnp.linalg.solve(S + 1e-8 * jnp.eye(d), jnp.eye(d))
    Wc = mu @ Si
    b = -0.5 * jnp.einsum("kd,kd->k", mu, Wc)
    return Wc, b


def predict(F: ArrayLike, Wc: ArrayLike, b: ArrayLike) -> jax.Array:
    return jnp.argmax(F @ Wc.T + b, axis=1)


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike, n_class: int) -> jax.Array:
    def cls_acc(k: int) -> jax.Array:
        m = y_true == k
        return jnp.where(m.sum() > 0, (jnp.where(m, y_pred == k, False).sum()) /
                         jnp.maximum(m.sum(), 1), jnp.nan)
    accs = jnp.array([cls_acc(k) for k in range(n_class)])
    return jnp.nanmean(accs)


def standardise(Ftr: ArrayLike, Fte: ArrayLike) -> tuple[jax.Array, jax.Array]:
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    return (Ftr - mu) / sd, (Fte - mu) / sd


def eval_fold(Xtr: ArrayLike, ytr: ArrayLike, Xte: ArrayLike, yte: ArrayLike,
              n_win: int, n_pca: int, n_class: int,
              res_tr: ArrayLike | None = None, res_te: ArrayLike | None = None) -> jax.Array:
    Ftr, Fte = windowed_riemann(Xtr, Xte, n_win, n_pca)
    if res_tr is not None:
        a, b = standardise(Ftr, Fte)
        c, d = standardise(res_tr, res_te)
        Ftr, Fte = jnp.hstack([a, c]), jnp.hstack([b, d])
    Ftr, Fte = standardise(Ftr, Fte)
    Wc, bb = shrinkage_lda_fit(Ftr, ytr, n_class)
    return balanced_accuracy(yte, predict(Fte, Wc, bb), n_class)
