"""
One pooled network for all subjects, with subject identity factored out.

Pooling is what makes a learned model possible at all: 3135 trials (285 per class) instead of
209 per subject. But pooling failed before (11-phoneme 0.121 vs 0.174 per-subject) because
subject identity and phoneme identity were entangled. So the architecture separates them:

    per-subject Riemannian whitening   removes the individual covariance baseline (0 params)
    shared spatial filters             learned across all 3135 trials
    multi-dilation temporal conv       several timescales inside the network
    subject embedding -> head          subject-specific decision offset
    gradient reversal -> subject head  forces the shared trunk to be subject-invariant

Comparisons:
    per-subject hand-crafted  (windowed Riemannian + means)      reference
    pooled, no factorisation                                     ablation
    pooled + whitening                                           ablation
    pooled + whitening + adversarial + subject embedding         full model

Evaluation is trial-level 5-fold, stratified by (subject, class), so every fold contains all
subjects -- the same protocol under which per-subject scored 0.174 and naive pooling 0.121.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power as fmp
from scipy.stats import ttest_rel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
BASE = (-0.2, -0.11)
DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")


def load_pooled(task: str, wmax: float, data_dir: str = "npz"):
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "sub-S*_ses-02.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    Xs, ys, sid = [], [], []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        X, t = d["X"].astype(float), d["times"]
        meta = pd.DataFrame({c: [str(v) for v in d[c]]
                             for c in ["task", "tms_target", "phoneme"]})
        k = ((meta.task == "single-phoneme")
             & meta.tms_target.str.startswith("control")).values
        bm = (t >= BASE[0]) & (t <= BASE[1])
        wm = (t >= 0.0) & (t <= wmax)
        Xb = (X[k] - X[k][:, :, bm].mean(2, keepdims=True))[:, :, wm]
        ph = meta.phoneme.values[k]
        m = np.isin(ph, STOPS) if task == "place" else np.isin(ph, PHON)
        y = (np.where(np.isin(ph[m], ["d", "t"]), 1, 0) if task == "place"
             else np.array([PHON.index(v) for v in ph[m]]))
        Xs.append(Xb[m]); ys.append(y); sid.append(np.full(m.sum(), i))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(sid), len(files))


def whiten_by_subject(X, sid, fit_idx):
    """Riemannian recentring: each subject's geometric mean covariance -> identity."""
    out = X.copy()
    for s in np.unique(sid):
        tr = fit_idx[sid[fit_idx] == s]
        if len(tr) < 8:
            continue
        C = Covariances("oas").fit_transform(X[tr])
        W = np.real(fmp(mean_riemann(C), -0.5))
        m = sid == s
        out[m] = np.einsum("ij,njt->nit", W, X[m])
    return out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    """Shared spatial filters + multi-dilation temporal convolutions, with a subject
    embedding on the head and a gradient-reversed subject classifier on the trunk."""

    def __init__(self, n_ch, n_time, n_cls, n_sub, f_spat=16, f_temp=12,
                 dilations=(1, 2, 4, 8), emb=8, use_emb=True, use_adv=True):
        super().__init__()
        self.use_emb, self.use_adv = use_emb, use_adv
        self.spatial = nn.Conv1d(n_ch, f_spat, 1, bias=False)
        self.bn_s = nn.BatchNorm1d(f_spat)
        self.branches = nn.ModuleList([
            nn.Conv1d(f_spat, f_temp, 15, padding=7 * d, dilation=d, groups=1)
            for d in dilations])
        self.bn_t = nn.BatchNorm1d(f_temp * len(dilations))
        self.pool = nn.AvgPool1d(8)
        feat = f_temp * len(dilations) * (n_time // 8)
        self.drop = nn.Dropout(0.5)
        self.emb = nn.Embedding(n_sub, emb)
        self.head = nn.Linear(feat + (emb if use_emb else 0), n_cls)
        self.sub_head = nn.Linear(feat, n_sub)

    def trunk(self, x):
        h = torch.relu(self.bn_s(self.spatial(x)))
        h = torch.cat([b(h) for b in self.branches], 1)
        h = torch.relu(self.bn_t(h))
        return self.drop(self.pool(h).flatten(1))

    def forward(self, x, s, lam=1.0):
        z = self.trunk(x)
        zc = torch.cat([z, self.emb(s)], 1) if self.use_emb else z
        logits = self.head(zc)
        slog = self.sub_head(GradReverse.apply(z, lam)) if self.use_adv else None
        return logits, slog


def run_fold(Xtr, ytr, str_, Xte, yte, ste, n_cls, n_sub, cfg, epochs=60, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-8
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEV)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEV)
    ya = torch.tensor(ytr, dtype=torch.long, device=DEV)
    sa = torch.tensor(str_, dtype=torch.long, device=DEV)
    sb = torch.tensor(ste, dtype=torch.long, device=DEV)
    net = Net(A.shape[1], A.shape[2], n_cls, n_sub,
              use_emb=cfg["emb"], use_adv=cfg["adv"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, bs = len(A), 128
    for ep in range(epochs):
        net.train()
        lam = cfg["lam"] * min(1.0, ep / max(1, epochs // 3))
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lo, sl = net(A[idx], sa[idx], lam)
            loss = F.cross_entropy(lo, ya[idx])
            if sl is not None:
                loss = loss + cfg["adv_w"] * F.cross_entropy(sl, sa[idx])
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = net(B, sb, 0.0)[0].argmax(1).cpu().numpy()
    return balanced_accuracy_score(yte, pred)


CFGS = {
    "pooled_plain":      dict(whiten=False, emb=False, adv=False, lam=0.0, adv_w=0.0),
    "pooled_whiten":     dict(whiten=True,  emb=False, adv=False, lam=0.0, adv_w=0.0),
    "pooled_wh_emb":     dict(whiten=True,  emb=True,  adv=False, lam=0.0, adv_w=0.0),
    "FULL_wh_emb_adv":   dict(whiten=True,  emb=True,  adv=True,  lam=1.0, adv_w=0.3),
}


def main(task, wmax, seeds):
    X, y, sid, n_sub = load_pooled(task, wmax)
    n_cls = len(np.unique(y))
    chance = 1.0 / n_cls
    print("task=%s  window 0-%.1fs  trials %d  classes %d  subjects %d  device %s\n" % (
        task, wmax, len(X), n_cls, n_sub, DEV))
    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid, y)])
    scores = {k: [] for k in CFGS}
    for sd in range(seeds):
        for fi, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat)):
            for name, cfg in CFGS.items():
                Xu = whiten_by_subject(X, sid, tr) if cfg["whiten"] else X
                a = run_fold(Xu[tr], y[tr], sid[tr], Xu[te], y[te], sid[te],
                             n_cls, n_sub, cfg, seed=sd)
                scores[name].append(a)
            print("  seed %d fold %d  %s" % (sd, fi, "  ".join(
                "%s %.3f" % (k[:12], scores[k][-1]) for k in CFGS)), flush=True)
    print("\n=== %s (chance %.3f) ===" % (task, chance))
    base = np.array(scores["pooled_plain"])
    for k in CFGS:
        a = np.array(scores[k])
        _, p = ttest_rel(a, base) if k != "pooled_plain" else (0, np.nan)
        print("  %-18s %.3f +- %.3f   vs plain %+.3f (p=%.4f)" % (
            k, a.mean(), a.std(), a.mean() - base.mean(), p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["place", "ph11"], default="ph11")
    ap.add_argument("--wmax", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    main(a.task, a.wmax, a.seeds)
