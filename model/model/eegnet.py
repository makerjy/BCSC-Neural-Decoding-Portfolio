
"""
EEGNet baseline, on the folds everything else in this session was scored on.

Two reasons to run it rather than cite it.

1. It is the standard reference for this dataset family, and every number produced today is
   from a covariance/tangent-space pipeline. If the classical route is leaving something on
   the table, a learned spatio-temporal filter bank is where it would show.

2. It tests the pooling conclusion from the opposite direction. The measurement today was
   that subjects' discriminant directions are near-orthogonal (cos 0.01 against a
   within-subject split-half of 0.45), so pooling cannot add information *to a linear
   readout*. A network trained on 2662 trials can learn shared filters that a per-subject
   linear model never could. If EEGNet-pooled beats EEGNet-per-subject, the conclusion has to
   be narrowed to "linear readouts cannot pool"; if it does not, the conclusion holds for
   learned representations too.

Arms, identical folds, balanced accuracy:
    persubj          one network per subject, that subject's ~166 training trials
    pooled           one network on all training trials, subject identity not given
    pooled+subjbias  same trunk, plus a per-subject additive logit bias (the network's
                     version of hier2's personal class means)
    pooled+ft_head   pooled trunk frozen, final layer refitted per subject -- the direct test
                     of "shared representation, personal readout"

Architecture is EEGNet-8,2 (Lawhern et al. 2018) with kernel length half the sampling rate,
max-norm constraints on the depthwise and dense layers, early stopping on an inner
validation split taken from the training folds only.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, copy
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, load_pooled, loso_folds)

class EEGNet(nn.Module):
    def __init__(self, n_cls, n_ch, n_t, F1=8, D=2, F2=16, kern=128, drop=0.25, n_sub=0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.depth = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.sep = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            d = self._trunk(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.head = nn.Linear(d, n_cls)
        self.subj = nn.Embedding(n_sub, n_cls) if n_sub else None
        if self.subj is not None:
            nn.init.zeros_(self.subj.weight)

    def _trunk(self, x):
        x = self.block2(self.depth(self.block1(x)))
        return self.sep(x).flatten(1)

    def forward(self, x, s=None):
        z = self.head(self._trunk(x))
        return z + self.subj(s) if (self.subj is not None and s is not None) else z

    def constrain(self):
        with torch.no_grad():
            self.depth.weight.data = torch.renorm(self.depth.weight.data, 2, 0, 1.0)
            self.head.weight.data = torch.renorm(self.head.weight.data, 2, 0, 0.25)


def adapt_bn(model, Xte, dev, passes=3, bs=64):
    """Recompute BatchNorm running statistics on the held-out subject's own trials.

    Uses their raw EEG and no labels, so it is legitimate for a new person and is the deep
    counterpart of the Riemannian recentring that gave the classical model +0.017 on LOSO.
    Without this the comparison is asymmetric: the classical model got an unlabelled
    adaptation and the network did not. momentum=None makes BN accumulate a true average
    over the passed data rather than an exponential trace.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            for i in range(0, len(Xte), bs):
                model(torch.as_tensor(Xte[i:i + bs], device=dev))
    model.eval()
    return model


def bal_acc(model, X, y, s, dev, use_subj):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.as_tensor(X[i:i + 512], device=dev)
            sb = torch.as_tensor(s[i:i + 512], device=dev) if use_subj else None
            out.append(model(xb, sb).argmax(1).cpu().numpy())
    return balanced_accuracy_score(y, np.concatenate(out))


def train(model, Xtr, ytr, str_, Xva, yva, sva, dev, use_subj, epochs, patience,
          lr=1e-3, bs=64, params=None):
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(Xtr, device=dev)
    yt = torch.as_tensor(ytr, device=dev)
    st = torch.as_tensor(str_, device=dev)
    best, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[j], st[j] if use_subj else None), yt[j])
            loss.backward()
            opt.step()
            model.constrain()
        v = bal_acc(model, Xva, yva, sva, dev, use_subj)
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best


def inner_split(y, sid, idx, frac, seed):
    strat = np.array(["%d_%d" % (a, b) for a, b in zip(sid[idx], y[idx])])
    n_sp = max(2, int(round(1 / frac)))
    tr, va = next(iter(StratifiedKFold(n_sp, shuffle=True,
                                       random_state=seed).split(idx, strat)))
    return idx[tr], idx[va]


def persubj_bal(pred, y, sid, te):
    """Per-subject balanced accuracy, averaged -- comparable to the sk_lda_persubj numbers."""
    out = []
    for s in np.unique(sid[te]):
        m = sid[te] == s
        if m.sum() >= len(np.unique(y)):
            out.append(balanced_accuracy_score(y[te][m], pred[m]))
    return float(np.mean(out))


def predict(model, X, te, sid, dev, use_subj):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            j = te[i:i + 512]
            xb = torch.as_tensor(X[j], device=dev)
            sb = torch.as_tensor(sid[j], device=dev) if use_subj else None
            out.append(model(xb, sb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def run_fold(X, y, sid, tr, te, n_cls, n_sub, dev, a, seed):
    torch.manual_seed(seed)
    n_ch, n_t = X.shape[2], X.shape[3]
    mu, sd = X[tr].mean(), X[tr].std()
    Xn = ((X - mu) / (sd + 1e-9)).astype(np.float32)
    itr, iva = inner_split(y, sid, tr, a.val_frac, seed)
    out = {}

    # ---- pooled trunk, no subject information
    m = EEGNet(n_cls, n_ch, n_t, kern=a.kern, drop=a.drop).to(dev)
    train(m, Xn[itr], y[itr], sid[itr], Xn[iva], y[iva], sid[iva], dev, False,
          a.epochs, a.patience, a.lr, a.bs)
    p = predict(m, Xn, te, sid, dev, False)
    out["pooled"] = balanced_accuracy_score(y[te], p)
    out["pooled_persubjmetric"] = persubj_bal(p, y, sid, te)

    # ---- same trunk, final layer refitted per subject
    ft = {}
    for s in range(n_sub):
        s_tr = itr[sid[itr] == s]
        s_va = iva[sid[iva] == s]
        if len(s_tr) < 3 * n_cls or len(s_va) < n_cls:
            continue
        ms = copy.deepcopy(m)
        for prm in ms.parameters():
            prm.requires_grad_(False)
        ms.head.weight.requires_grad_(True); ms.head.bias.requires_grad_(True)
        train(ms, Xn[s_tr], y[s_tr], sid[s_tr], Xn[s_va], y[s_va], sid[s_va], dev, False,
              a.ft_epochs, a.patience, a.ft_lr, a.bs,
              params=[ms.head.weight, ms.head.bias])
        ft[s] = ms
    pf = np.empty(len(te), int)
    for s in np.unique(sid[te]):
        pos = np.where(sid[te] == s)[0]
        mm = ft.get(s, m)
        pf[pos] = predict(mm, Xn, te[pos], sid, dev, False)
    out["pooled+ft_head"] = balanced_accuracy_score(y[te], pf)
    out["pooled+ft_head_persubjmetric"] = persubj_bal(pf, y, sid, te)

    # ---- pooled trunk with an additive per-subject logit bias
    m2 = EEGNet(n_cls, n_ch, n_t, kern=a.kern, drop=a.drop, n_sub=n_sub).to(dev)
    train(m2, Xn[itr], y[itr], sid[itr], Xn[iva], y[iva], sid[iva], dev, True,
          a.epochs, a.patience, a.lr, a.bs)
    p2 = predict(m2, Xn, te, sid, dev, True)
    out["pooled+subjbias"] = balanced_accuracy_score(y[te], p2)
    out["pooled+subjbias_persubjmetric"] = persubj_bal(p2, y, sid, te)

    # ---- one network per subject
    ps = np.empty(len(te), int)
    for s in np.unique(sid[te]):
        pos = np.where(sid[te] == s)[0]
        s_tr = itr[sid[itr] == s]
        s_va = iva[sid[iva] == s]
        if len(s_tr) < 3 * n_cls or len(s_va) < n_cls:
            ps[pos] = predict(m, Xn, te[pos], sid, dev, False)
            continue
        ms = EEGNet(n_cls, n_ch, n_t, kern=a.kern, drop=a.drop).to(dev)
        train(ms, Xn[s_tr], y[s_tr], sid[s_tr], Xn[s_va], y[s_va], sid[s_va], dev, False,
              a.epochs, a.patience, a.lr, a.bs)
        ps[pos] = predict(ms, Xn, te[pos], sid, dev, False)
    out["persubj"] = balanced_accuracy_score(y[te], ps)
    out["persubj_persubjmetric"] = persubj_bal(ps, y, sid, te)
    return out


ARMS = ["persubj", "pooled", "pooled+subjbias", "pooled+ft_head"]
ARMS_LOSO = ["pooled", "pooled+bn_adapt"]


def run_fold_loso(X, y, sid, tr, te, n_cls, n_sub, dev, a, seed):
    """Held-out subject: no labelled trials exist for them, so only the shared trunk applies."""
    torch.manual_seed(seed)
    mu, sd = X[tr].mean(), X[tr].std()
    Xn = ((X - mu) / (sd + 1e-9)).astype(np.float32)
    itr, iva = inner_split(y, sid, tr, a.val_frac, seed)
    m = EEGNet(n_cls, X.shape[2], X.shape[3], kern=a.kern, drop=a.drop).to(dev)
    train(m, Xn[itr], y[itr], sid[itr], Xn[iva], y[iva], sid[iva], dev, False,
          a.epochs, a.patience, a.lr, a.bs)
    p = predict(m, Xn, te, sid, dev, False)
    out = {"pooled": balanced_accuracy_score(y[te], p)}
    out["pooled_persubjmetric"] = out["pooled"]
    ma = adapt_bn(copy.deepcopy(m), Xn[te], dev)          # unlabelled, test subject only
    pa = predict(ma, Xn, te, sid, dev, False)
    out["pooled+bn_adapt"] = balanced_accuracy_score(y[te], pa)
    out["pooled+bn_adapt_persubjmetric"] = out["pooled+bn_adapt"]
    return out


def main(a):
    Xr, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data)
    X = Xr[:, None].astype(np.float32)
    y = y.astype(np.int64); sid = sid.astype(np.int64)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    dev = torch.device(a.device)
    print("task=%s trials=%d subjects=%d classes=%d shape=%s device=%s | EEGNet-8,2 "
          "kern=%d drop=%.2f epochs<=%d patience=%d" % (
              a.task, len(X), n_sub, n_cls, tuple(X.shape[1:]), dev,
              a.kern, a.drop, a.epochs, a.patience), flush=True)

    global ARMS
    loso = a.split == "loso"
    if loso:
        ARMS = ARMS_LOSO
    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    R = {k: [] for k in ARMS}
    RP = {k: [] for k in ARMS}
    if loso:
        folds = loso_folds(sid)
        for fi, (tr, te) in enumerate(folds):
            o = run_fold_loso(X, y, sid, tr, te, n_cls, n_sub, dev, a, fi)
            for k in ARMS:
                R[k].append(o[k]); RP[k].append(o[k + "_persubjmetric"])
            print("  held-out subject %2d  pooled %.3f  +bn_adapt %.3f"
                  % (fi, o["pooled"], o["pooled+bn_adapt"]), flush=True)
    else:
        for sd in range(a.seeds):
            for fi, (tr, te) in enumerate(
                    StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat)):
                o = run_fold(X, y, sid, tr, te, n_cls, n_sub, dev, a, sd * 10 + fi)
                for k in ARMS:
                    R[k].append(o[k]); RP[k].append(o[k + "_persubjmetric"])
                print("  seed %d fold %d  " % (sd, fi) + "  ".join(
                    "%s %.3f/%.3f" % (k[:14], o[k], o[k + "_persubjmetric"])
                    for k in ARMS), flush=True)

    print()
    print("=== %s (chance %.3f)  EEGNet ===" % (a.task, chance))
    print("%-18s %10s %10s" % ("arm", "overall", "per-subj-avg"))
    for k in ARMS:
        print("%-18s %10.3f %10.3f" % (k, np.mean(R[k]), np.mean(RP[k])))
    print()
    if loso and len(ARMS) == 2:
        A = np.array(R["pooled"], float); B = np.array(R["pooled+bn_adapt"], float)
        _, pp = ttest_rel(B, A)
        print()
        print("  bn_adapt - pooled  %+.4f   folds up %d/%d   paired p=%.4f"
              % ((B - A).mean(), int((B > A).sum()), len(A), pp))
    ref = np.array(R[ARMS[0]], float)
    for k in sorted(ARMS, key=lambda k: -np.mean(R[k])):
        v = np.array(R[k], float)
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-18s %.3f  vs-chance p=%.7f  vs-persubj %+.3f (p=%.4f) %s" % (
            k, v.mean(), p0, v.mean() - ref.mean(), pb,
            "**BEATS**" if pb < 0.05 and v.mean() > ref.mean() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--split", default="trial", choices=["trial", "loso"])
    ap.add_argument("--kern", type=int, default=128)
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--ft_epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    main(ap.parse_args())
