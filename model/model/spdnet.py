
"""
The connectionist arm, inside the same Riemannian geometry.

`final_model.py` reaches its representation by closed form: a supervised generalised
eigendecomposition picks the spatial projection, a hand-specified set of rows builds one SPD
object per trial, and a fixed log-Euclidean map flattens it. The only thing gradient descent
touches in this repo is `eegnet.py`, which is a Euclidean CNN -- so the existing comparison
("designed 0.306 beats EEGNet 0.225") confounds two differences at once: designed-vs-learned
AND SPD-manifold-vs-Euclidean. This file removes the second confound.

Every arm below sees the same trials and ends in the same geometry. What changes is how much
of the representation is *specified* and how much *emerges* from backpropagation:

    designed        snr_filter -> unified_spd -> fixed log-Euclidean tangent -> LDA
                    (this is final_model.py, recomputed here on identical folds)
    tangent_lin     same representation, softmax head trained by backprop
                    -- isolates "LDA vs a learned linear head" so it cannot be mistaken
                       for the effect of the manifold layers
    spdnet          same SPD input, then BiMap -> ReEig -> BiMap -> ReEig -> LogEig learned
                    by backprop (Huang & Van Gool 2017)
                    -- the representation emerges, the front end stays designed
    spdnet_e2e      a learned spatial projection replaces snr_filter as well, and the SPD
                    object is built differentiably, so nothing but the *form* of the object
                    is specified by hand

Two scores per net arm, because readout and representation must not be confounded:

    softmax   the network's own head, per-subject balanced accuracy
    +lda      the LogEig features handed to sdcore.lda_persubj -- byte-identical readout to
              `designed`, so a difference here is a difference in representation alone

Why this is worth running even though it is likely to lose. The session's conclusion is that
emergence needs samples this dataset does not have (208 trials/subject, 11 classes, ~19 per
class per subject; the learning curve is still on its steep part). That conclusion is
currently supported only by a Euclidean CNN. If a manifold network -- which is far more
sample-efficient, because BiMap layers are small and the SPD structure is given rather than
learned -- also fails to beat the closed-form filter, the claim generalises from "deep nets
need data" to "learning the representation does not pay at this sample size, even when the
geometry is handed to it". If it wins, the designed model is leaving something on the table
and we would want to know that before the poster goes up.

Results, ph11, 15 folds, per-subject balanced accuracy, chance 0.091 (logs/spdnet_ph11.log):

    rand_rot                0.255   +0.007 vs designed  CI [+0.001,+0.014] p=0.035  11/15
    designed                0.248        --
    spdnet      +classical  0.232   -0.016              CI [-0.026,-0.007] p=0.003   2/15
    spdnet_e2e  +classical  0.222   -0.026              CI [-0.037,-0.014] p=0.0003  1/15
    spdnet      softmax     0.180   -0.067
    spdnet_e2e  softmax     0.180   -0.068

So: learning the representation loses, learning the front end as well loses more, and the
network's own pooled head loses heavily to a per-subject readout on its own features -- which
is the pooled-vs-personal result of this session reproduced inside a learned model.

`rand_rot` is there because of a prediction that failed, and the failure is the interesting
part. For a *square* BiMap, Q is orthogonal, so log(Q^T C Q) = Q^T log(C) Q, and the
sqrt(2)-weighted upper triangle makes that an orthogonal transform of the feature vector. A
classifier invariant to invertible linear maps cannot see it, so `--ladder 36` was expected to
reproduce `designed`. It did not -- it beat it (0.256 vs 0.248, p=0.042, 11/15). The reason is
that `lda_persubj` is *not* invariant: StandardScaler rescales each coordinate and Ledoit-Wolf
shrinks toward the resulting identity, so the basis decides which directions get regularised.

`rand_rot` applies a fixed random orthogonal rotation and learns nothing. It captures the
whole effect (logs/spdnet_rot36_ph11.log, 15 folds, identical folds throughout):

    designed 0.248  ->  random rotation 0.255  ->  learned rotation 0.256
      random  - designed   +0.0073  CI [+0.001,+0.014]  p=0.035  11/15
      learned - designed   +0.0077  CI [+0.000,+0.015]  p=0.042  11/15
      learned - random     +0.0005  CI [-0.004,+0.005]  p=0.84    7/15   -- nothing

Backpropagation on a dimension-preserving manifold layer contributes nothing beyond a random
rotation, and the interval is tight enough (+/-0.005) to say so rather than to fail to reject.
The gain belongs to the basis, not to learning. This is the same shape as the Laplacian-basis
result in SUMMARY 4.2 -- the gain turned out to be scale alignment rather than anatomy -- and
it is a finding about the *designed* model: its tangent basis is mildly badly conditioned for
a shrinkage regulariser, and almost any rotation repairs it.

(An earlier 5-fold pass put learned - random at +0.008, p=0.099. That was noise; the 15-fold
interval excludes it. Recorded because the 5-fold number is the one that would have been
believed if the run had been stopped when it looked interesting.)

One asymmetry, kept deliberately because `eegnet.py` has it too: the learned arms hold out
`--val_frac` of the training fold for early stopping, so they fit on ~20% fewer trials than
`designed` does. On a dataset whose learning curve is still rising this is not free, and it is
part of what "needs more data" means rather than a flaw in the comparison.

Numerics: `--selftest` checks the torch SPD construction against sdcore element by element
before any training happens. The eigendecomposition backward has 1/(l_i - l_j) terms, so the
inputs are kept well conditioned (unified_spd already shrinks 0.08 toward a scaled identity,
and BiMap keeps orthonormal columns so eigenvalues interlace and cannot collapse to zero) and
a small jitter is added before every eigh.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from eegnet import inner_split, persubj_bal
from sdcore import (CHANCE, DATA, folds_for, lda_persubj, lda_pooled, load_pooled, report,
                    snr_filter, spd_log, tangent, triu, unified_spd)

JITTER = 1e-5


# ---------------------------------------------------------------- SPD layers

def _sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


def _eigh(C):
    """eigh with a jitter floor, scaled to the matrix.

    The jitter has to be *relative*: these objects carry eigenvalues from 1e0 to 1e4 depending
    on the phi scaling, so a fixed 1e-5 would be a no-op on one trial and dominant on another.
    Scaling by trace/k makes it a fixed fraction of the mean eigenvalue instead.

    The CPU float64 retry is not decoration. cusolver's batched syevd returns error 37 on
    inputs it considers too ill-conditioned, and it does so rarely enough that discovering it
    mid-run would cost a whole sweep.
    """
    k = C.shape[-1]
    S = _sym(C)
    eye = torch.eye(k, dtype=C.dtype, device=C.device)
    # detached on purpose: the jitter is a conditioning constant, not part of the function
    # being differentiated. _EigFn's backward assumes exactly this.
    tc = S.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].detach() / k
    S = S + JITTER * tc * eye
    try:
        return torch.linalg.eigh(S)
    except Exception:
        w, U = torch.linalg.eigh(S.double().cpu())
        return w.to(C.dtype).to(C.device), U.to(C.dtype).to(C.device)


class _EigFn(torch.autograd.Function):
    """Apply a scalar function to the eigenvalues of a symmetric matrix, with a backward that
    survives repeated eigenvalues.

    The derivative of U f(L) U^T is U (P . (U^T G U)) U^T where P is the Loewner matrix,
    P_ij = (f(l_i) - f(l_j)) / (l_i - l_j) off the diagonal and f'(l_i) on it. torch's generic
    eigh backward forms that quotient unconditionally, so two eigenvalues that coincide give
    inf and then NaN -- which is exactly what killed a square-BiMap run at epoch 17. Here the
    quotient is replaced by its limit f'(l) whenever the gap closes, which is not a fudge but
    the value the quotient converges to.
    """

    @staticmethod
    def forward(ctx, S, kind, eps, gap_rel):
        w, U = _eigh(S)
        if kind == "log":
            wc = w.clamp_min(eps)
            fw = wc.log()
            dfw = torch.where(w > eps, 1.0 / wc, torch.zeros_like(w))
        elif kind == "re":
            fw = w.clamp_min(eps)
            dfw = (w > eps).to(w.dtype)
        else:
            raise ValueError(kind)
        ctx.save_for_backward(w, U, fw, dfw)
        ctx.gap = gap_rel
        return U @ torch.diag_embed(fw) @ U.transpose(-1, -2)

    @staticmethod
    def backward(ctx, G):
        w, U, fw, dfw = ctx.saved_tensors
        G = _sym(G)
        Gt = U.transpose(-1, -2) @ G @ U
        d = w.unsqueeze(-1) - w.unsqueeze(-2)
        gap = ctx.gap * w.abs().amax(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-12)
        close = d.abs() < gap
        quot = (fw.unsqueeze(-1) - fw.unsqueeze(-2)) / torch.where(close, torch.ones_like(d), d)
        lim = 0.5 * (dfw.unsqueeze(-1) + dfw.unsqueeze(-2))
        P = torch.where(close, lim, quot)
        return U @ (P * Gt) @ U.transpose(-1, -2), None, None, None


class BiMap(nn.Module):
    """C -> W^T C W with W orthonormal, so the output stays SPD and eigenvalues interlace.

    Q is kept on the Stiefel manifold by torch's own parametrisation rather than by
    orthonormalising a free parameter, and both of the obvious hand-rolled alternatives were
    tried and rejected on measurements:

      QR   with a square W the smallest singular value of W collapsed from 0.79 to 3e-3
           inside 25 epochs, and QR on a near-singular matrix returns trailing columns that
           are numerical noise -- which is what made cusolver's eigh fail to converge
           downstream (error 37).
      SVD  the polar factor Q = U V^T is stable in the forward direction, but its backward
           carries 1/(s_i^2 - s_j^2), and an orthogonal initialisation makes every singular
           value exactly 1, so the very first backward pass is NaN.

    `parametrizations.orthogonal` sidesteps both: it maps an unconstrained parameter through a
    matrix exponential (square) or a Householder product (rectangular), neither of which has a
    denominator that can vanish. Weight decay on the underlying tensor is still meaningless
    here, and `train` excludes it.
    """

    def __init__(self, n_in, n_out):
        super().__init__()
        lin = nn.Linear(n_in, n_out, bias=False)      # weight is (n_out, n_in)
        self.lin = torch.nn.utils.parametrizations.orthogonal(lin)

    def forward(self, C):
        Q = self.lin.weight                            # orthonormal rows, Q Q^T = I
        return _sym(Q @ C @ Q.transpose(0, 1))


class ReEig(nn.Module):
    """The manifold's nonlinearity: rectify eigenvalues at eps. This is the layer the fixed
    pipeline has no counterpart for -- `designed` applies exactly one nonlinearity (the matrix
    log) and it is not learned."""

    def __init__(self, eps=1e-4, gap=1e-6):
        super().__init__()
        self.eps, self.gap = eps, gap

    def forward(self, C):
        return _sym(_EigFn.apply(_sym(C), "re", self.eps, self.gap))


class LogEig(nn.Module):
    """log-Euclidean map to the tangent space, then the upper triangle with the off-diagonal
    weighted by sqrt(2) -- the same convention as sdcore.triu, so the feature vectors are
    directly comparable to the classical ones."""

    def __init__(self, eps=1e-8, gap=1e-6):
        super().__init__()
        self.eps, self.gap = eps, gap

    def forward(self, C):
        L = _sym(_EigFn.apply(_sym(C), "log", self.eps, self.gap))
        d = L.shape[-1]
        iu = torch.triu_indices(d, d, device=L.device)
        diag = iu[0] == iu[1]
        wt = torch.where(diag, torch.ones_like(diag, dtype=L.dtype),
                         torch.full_like(diag, np.sqrt(2.0), dtype=L.dtype))
        return L[..., iu[0], iu[1]] * wt


class SubjectCentre(nn.Module):
    """Per-subject mean subtraction in tangent space, estimated on training trials only.

    The classical pipeline does this inside `sdcore.tangent`, and it is worth a lot here
    because subjects' discriminant directions are near-orthogonal. It uses no labels, so a
    held-out subject may use its own trials (Riemannian recentring, Zanini et al. 2018).
    The centre is a constant refreshed between epochs, not a parameter -- the same status it
    has in the closed-form pipeline.
    """

    def __init__(self, n_sub, dim):
        super().__init__()
        self.register_buffer("centre", torch.zeros(n_sub, dim))

    def forward(self, F, sid):
        return F - self.centre[sid]


class SpatialProj(nn.Module):
    """Learned 61 -> d spatial projection, the emergent counterpart of snr_filter. Left
    unconstrained because snr_filter is unconstrained too (generalised eigenvectors are not
    orthonormal). A global rescaling of this layer is harmless: it shifts log(C) by a multiple
    of the identity, which SubjectCentre and the head bias absorb."""

    def __init__(self, n_ch, d, init=None):
        super().__init__()
        W = torch.as_tensor(init, dtype=torch.float32).clone() if init is not None \
            else torch.linalg.qr(torch.randn(n_ch, d))[0]
        self.W = nn.Parameter(W)

    def forward(self, X):
        return torch.einsum("ck,nct->nkt", self.W, X)


# ---------------------------------------------------------------- differentiable SPD build

def phi_rows_t(T, K, device, dtype):
    t = (torch.arange(T, device=device, dtype=dtype) + 0.5) / T
    P = torch.stack([torch.cos(np.pi * k * t) for k in range(K + 1)])
    return P / P.pow(2).mean(1, keepdim=True).sqrt()


def phi_scale_from(Z, tr, p, tau):
    """sdcore takes the phi row scale from the delay-augmented training trials. The learned
    arms must use the same constant rather than a per-minibatch estimate, or the geometry of
    the object would drift with the batch."""
    T = Z.shape[2]
    span = (p - 1) * tau
    A = np.concatenate([Z[tr][:, :, i * tau:T - span + i * tau] for i in range(p)], 1) \
        if p > 1 else Z[tr]
    return float(np.sqrt((A ** 2).mean()))


def unified_spd_t(Z, p, tau, K, phi_scale=1.0, shrink=0.08, scale=None):
    """torch port of sdcore.unified_spd, verified element by element in --selftest.

    `scale` is the phi row scaling. sdcore takes it from the training trials; passing it in
    keeps that behaviour instead of letting it drift with the minibatch.
    """
    if p > 1:
        T0 = Z.shape[2]
        span = (p - 1) * tau
        A = torch.cat([Z[:, :, i * tau:T0 - span + i * tau] for i in range(p)], 1)
    else:
        A = Z
    n, _, T = A.shape
    P = phi_rows_t(T, K, A.device, A.dtype)
    s = float(scale) if scale is not None else float(A.detach().pow(2).mean().sqrt())
    A = torch.cat([A, (P * (phi_scale * s)).expand(n, -1, -1)], 1)
    k = A.shape[1]
    C = A @ A.transpose(1, 2) / T
    tc = C.diagonal(dim1=1, dim2=2).sum(1)[:, None, None] / k
    eye = torch.eye(k, dtype=C.dtype, device=C.device)
    return (1 - shrink) * C + shrink * tc * eye


# ---------------------------------------------------------------- the network

class SPDNet(nn.Module):
    """BiMap/ReEig stack -> LogEig -> subject centring -> linear head.

    `dims` is the BiMap ladder, e.g. [36, 24, 16]. `dims=[36]` is the degenerate case with no
    manifold layers at all, which is exactly `tangent_lin`: the fixed log-Euclidean map
    followed by a learned linear head.
    """

    def __init__(self, dims, n_cls, n_sub, proj=None, spd_args=None):
        super().__init__()
        self.proj = proj
        self.spd_args = spd_args
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [BiMap(a, b), ReEig()]
        self.stack = nn.Sequential(*layers)
        self.logeig = LogEig()
        d = dims[-1]
        self.dim = d * (d + 1) // 2
        self.centre = SubjectCentre(n_sub, self.dim)
        self.head = nn.Linear(self.dim, n_cls)

    def features(self, inp, sid):
        C = unified_spd_t(self.proj(inp), **self.spd_args) if self.proj is not None else inp
        return self.centre(self.logeig(self.stack(C)), sid)

    def forward(self, inp, sid):
        return self.head(self.features(inp, sid))


# ---------------------------------------------------------------- train / evaluate

def _batched(model, inp, sid, idx, dev, bs=512, grad=False):
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            out.append(model.features(inp[j].to(dev), torch.as_tensor(sid[j], device=dev)))
    return torch.cat(out)


def centre_refs(sid, tr):
    """Which trials estimate each subject's tangent centre.

    A training subject uses its training trials. A held-out (LOSO) subject has none, so it
    uses its own trials -- no labels are touched either way, so this is the same unlabelled
    recentring `sdcore.tangent(adapt="unlabeled")` performs. Without it the held-out subject
    would enter uncentred while everyone else was centred, which is a split artefact rather
    than a property of the model.
    """
    out = {}
    for s in np.unique(sid):
        own_tr = tr[sid[tr] == s]
        out[int(s)] = own_tr if len(own_tr) >= 10 else np.where(sid == s)[0]
    return out


def refresh_centres(model, inp, sid, refs, dev):
    """Per-subject tangent centres, recomputed between epochs because the representation they
    centre is still moving. Held as a constant buffer, never a parameter -- the same status
    the reference point has in the closed-form pipeline."""
    model.eval()
    with torch.no_grad():
        model.centre.centre.zero_()
        for s, j in refs.items():
            model.centre.centre[s] = _batched(model, inp, sid, j, dev).mean(0)


def train_optimiser(model, lr=1e-3, wd=1e-4):
    """BiMap parameters are exempt from weight decay: they are pre-images of an orthogonal
    parametrisation, so shrinking them toward zero does not shrink anything the model uses."""
    bimap = {id(p) for m in model.modules() if isinstance(m, BiMap) for p in m.parameters()}
    return torch.optim.Adam(
        [{"params": [p for p in model.parameters() if id(p) not in bimap],
          "weight_decay": wd},
         {"params": [p for p in model.parameters() if id(p) in bimap],
          "weight_decay": 0.0}], lr=lr)


def train(model, inp, y, sid, tr, va, refs, dev, epochs, patience, lr, bs, wd):
    opt = train_optimiser(model, lr, wd)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        refresh_centres(model, inp, sid, refs, dev)
        model.train()
        perm = tr[np.random.permutation(len(tr))]
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            logit = model(inp[j].to(dev), torch.as_tensor(sid[j], device=dev))
            loss = lossf(logit, torch.as_tensor(y[j], device=dev))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss at epoch %d" % ep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(inp[va].to(dev),
                       torch.as_tensor(sid[va], device=dev)).argmax(1).cpu().numpy()
        v = balanced_accuracy_score(y[va], pv)
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    refresh_centres(model, inp, sid, refs, dev)
    return best


def run_net(dims, inp, y, sid, tr, te, n_cls, n_sub, dev, a, seed, proj=None, spd_args=None,
            loso=False):
    """Returns (softmax per-subject balanced acc, classical readout on the same features, dim).

    The second score exists so representation and readout are never confounded: it is the
    identical `sdcore` readout `designed` uses, fed the network's own tangent vectors.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SPDNet(dims, n_cls, n_sub, proj, spd_args).to(dev)
    itr, iva = inner_split(y, sid, tr, a.val_frac, seed)
    refs = centre_refs(sid, itr)
    train(model, inp, y, sid, itr, iva, refs, dev, a.epochs, a.patience, a.lr, a.bs, a.wd)

    model.eval()
    with torch.no_grad():
        pred = model(inp[te].to(dev),
                     torch.as_tensor(sid[te], device=dev)).argmax(1).cpu().numpy()
    soft = persubj_bal(pred, y, sid, te)

    F = _batched(model, inp, sid, np.arange(len(y)), dev).cpu().numpy()
    cls = lda_pooled(F, y, tr, te) if loso \
        else lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
    return soft, cls, model.dim


# ---------------------------------------------------------------- self test

def selftest(a):
    """The torch SPD path must reproduce sdcore before any conclusion is drawn from it."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((40, 14, 256))
    tr = np.arange(40)
    ok = True

    scale = phi_scale_from(Z, tr, a.p, a.tau)
    C_np = unified_spd(Z, tr, a.p, a.tau, a.K, a.phi)
    C_t = unified_spd_t(torch.as_tensor(Z, dtype=torch.float64), a.p, a.tau, a.K, a.phi,
                        scale=scale).numpy()
    d1 = np.abs(C_np - C_t).max()
    print("  unified_spd      torch vs sdcore   max|diff| = %.3e" % d1)
    ok &= d1 < 1e-9

    F_np = triu(spd_log(C_np))
    F_t = LogEig()(torch.as_tensor(C_np, dtype=torch.float64)).numpy()
    d2 = np.abs(F_np - F_t).max()
    print("  LogEig + triu    torch vs sdcore   max|diff| = %.3e   (jitter %.0e)" % (d2, JITTER))
    ok &= d2 < 1e-4

    # BiMap is checked in float32 because that is the dtype training runs in, and because
    # torch's orthogonal parametrisation keeps its trivialisation base in float32 whatever
    # the module is cast to -- so 1e-7 is the floor here, not a defect. A broken
    # orthonormalisation misses by ~1e-1, so 1e-5 still separates the two cases.
    side_np = C_np.shape[-1]
    C32 = torch.as_tensor(C_np, dtype=torch.float32)
    for n_out, tag in [(16, "tall"), (side_np, "square")]:
        bm = BiMap(side_np, n_out)
        Q = bm.lin.weight.detach()
        orth = float((Q @ Q.T - torch.eye(n_out)).abs().max())
        out = bm(C32)
        asym = float((out - out.transpose(-1, -2)).abs().max())
        lmin = float(torch.linalg.eigvalsh(out.double()).min())
        print("  BiMap %-7s  |QQ'-I| %.3e   asymmetry %.3e   min eigenvalue %.3e"
              % (tag, orth, asym, lmin))
        ok &= orth < 1e-5 and asym < 1e-4 and lmin > 0

    # a square BiMap is a similarity transform, so it must not move the eigenvalues at all --
    # this is why the manifold layers are invisible to a linearly-invariant readout unless
    # they reduce dimension
    sq = BiMap(side_np, side_np)
    e0 = torch.linalg.eigvalsh(C32.double())
    e1 = torch.linalg.eigvalsh(sq(C32).double())
    d3 = float((e0 - e1).abs().max() / e0.abs().max())
    print("  square BiMap     eigenvalue drift (relative)  %.3e" % d3)
    ok &= d3 < 1e-5

    # the custom Loewner backward must agree with torch's autograd where torch is right
    # (well-separated eigenvalues) and stay finite where torch is not (repeated ones)
    def naive(S, kind, eps):
        """Same forward, differentiated by torch instead of by _EigFn.backward."""
        w, U = _eigh(S)
        fw = w.clamp_min(eps).log() if kind == "log" else w.clamp_min(eps)
        return U @ torch.diag_embed(fw) @ U.transpose(-1, -2)

    g0 = rng.standard_normal((4, 12, 40))
    S_sep = torch.as_tensor(np.einsum("nit,njt->nij", g0, g0) / 40 + 0.5 * np.eye(12),
                            dtype=torch.float64)
    S_deg = torch.eye(12, dtype=torch.float64).expand(4, 12, 12).clone()
    for kind, epsv in [("log", 1e-8), ("re", 1e-4)]:
        a1 = S_sep.clone().requires_grad_(True)
        a2 = S_sep.clone().requires_grad_(True)
        (_EigFn.apply(a1, kind, epsv, 1e-6) * S_sep).sum().backward()
        (naive(a2, kind, epsv) * S_sep).sum().backward()
        dg = float((a1.grad - a2.grad).abs().max())
        b = S_deg.clone().requires_grad_(True)
        (_EigFn.apply(b, kind, epsv, 1e-6) * S_deg).sum().backward()
        fin = bool(torch.isfinite(b.grad).all())
        print("  _EigFn %-4s     vs autograd max|dgrad| %.3e | repeated-eig grad finite %s"
              % (kind, dg, fin))
        ok &= dg < 1e-8 and fin

    net = SPDNet([side_np, 24, 16], 11, 16)
    nn.CrossEntropyLoss()(net(C32[:8], torch.zeros(8, dtype=torch.long)),
                          torch.arange(8) % 11).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    gmax = float(max(g.abs().max() for g in grads))
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    print("  backward         %d grad tensors, all-finite %s, max|grad| = %.3e"
          % (len(grads), finite, gmax))
    ok &= finite and gmax > 0

    # the optimiser split must survive a rename of BiMap's internals -- it did not, once
    opt = train_optimiser(net, wd=1e-4)
    n_decay = sum(len(g["params"]) for g in opt.param_groups if g["weight_decay"] > 0)
    n_free = sum(len(g["params"]) for g in opt.param_groups if g["weight_decay"] == 0)
    print("  optimiser split  %d decayed, %d BiMap params exempt" % (n_decay, n_free))
    ok &= n_free == 2 and n_decay == len(list(net.parameters())) - 2

    print("\n  SELFTEST %s" % ("PASSED" if ok else "FAILED"))
    return ok


# ---------------------------------------------------------------- main

def main(a):
    if a.selftest:
        raise SystemExit(0 if selftest(a) else 1)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    n_cls = len(np.unique(y)); chance = CHANCE[a.task]
    side = a.p * a.d + a.K + 1
    print("task=%s trials=%d subjects=%d classes=%d | d=%d p=%d tau=%d K=%d | object side %d "
          "-> tangent %d | device=%s" % (a.task, len(X), n_sub, n_cls, a.d, a.p, a.tau, a.K,
                                         side, side * (side + 1) // 2, dev), flush=True)
    print("  arms: %s" % " ".join(a.arms), flush=True)

    folds = folds_for(a.split, X, y, tok, sid, a.seeds)
    res, dims = {}, {}
    Xt = torch.as_tensor(X, dtype=torch.float32)

    loso = a.split == "loso"
    for fi, (tr, te) in enumerate(folds):
        W = snr_filter(X, y, sid, tr, n_cls, a.d, a.eps, a.den)
        Z = np.einsum("ck,nct->nkt", W, X)
        C = unified_spd(Z, tr, a.p, a.tau, a.K, a.phi)
        Ct = torch.as_tensor(C, dtype=torch.float32)
        spd_args = dict(p=a.p, tau=a.tau, K=a.K, phi_scale=a.phi,
                        scale=phi_scale_from(Z, tr, a.p, a.tau))
        line = []

        def classical(Cin, tag):
            F = tangent(Cin, tr, sid, "unlabeled" if loso else "train")
            v = lda_pooled(F, y, tr, te) if loso \
                else lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
            res.setdefault(tag, []).append(v)
            dims[tag] = F.shape[1]
            line.append("%s %.3f" % (tag, v))

        if "designed" in a.arms:
            classical(C, "designed")

        if "rand_rot" in a.arms:
            # The control a learned square BiMap needs. A square BiMap is a similarity
            # transform: it cannot change the eigenvalues, so it adds no information, and the
            # tangent vectors it produces are an orthogonal transform of the designed ones.
            # Yet lda_persubj is not rotation invariant -- StandardScaler rescales each
            # coordinate and Ledoit-Wolf shrinks toward the resulting identity -- so *any*
            # rotation changes which directions get regularised. If a random rotation moves
            # the score as much as the learned one, the gain is a badly conditioned basis,
            # not something backpropagation found.
            rng = np.random.default_rng(1000 + fi)
            Q = np.linalg.qr(rng.standard_normal((C.shape[-1], C.shape[-1])))[0]
            classical(np.matmul(Q.T, np.matmul(C, Q)), "rand_rot")

        for name, ladder, proj_kind in [
                ("tangent_lin", [side], None),
                ("spdnet", [side] + a.ladder, None),
                ("spdnet_e2e", [side] + a.ladder, "learn"),
                ("spdnet_e2e_init", [side] + a.ladder, "snr")]:
            if name not in a.arms:
                continue
            if proj_kind is None:
                inp, proj, sa = Ct, None, None
            else:
                inp = Xt
                proj = SpatialProj(X.shape[1], a.d, W if proj_kind == "snr" else None)
                sa = spd_args
            s, c, dim = run_net(ladder, inp, y, sid, tr, te, n_cls, n_sub, dev, a, a.seed,
                                proj, sa, loso)
            res.setdefault(name + "|softmax", []).append(s)
            res.setdefault(name + "|+classical", []).append(c)
            dims[name] = dim
            line.append("%s %.3f/%.3f" % (name, s, c))

        print("  fold %d  %s" % (fi, "  ".join(line)), flush=True)

    print()
    print("=== %s (chance %.3f)  split=%s  %d folds ===" % (a.task, chance, a.split, len(folds)))
    for k in dims:
        print("  %-18s tangent dim %d" % (k, dims[k]))
    print()
    report(res, chance, ref="designed", label="per-subject balanced accuracy", width=24)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", default="trial", choices=["trial", "token", "time", "loso"])
    ap.add_argument("--arms", nargs="+",
                    default=["designed", "tangent_lin", "spdnet", "spdnet_e2e"])
    ap.add_argument("--ladder", nargs="+", type=int, default=[24, 16])
    ap.add_argument("--selftest", action="store_true")
    # representation, kept at final_model.py's values so the comparison is like for like
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--den", default="noise", choices=["eye", "noise"])
    ap.add_argument("--eps", type=float, default=0.05)
    # optimisation
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
