s = open("pooled_factor.py").read()
head = s[:s.index("def _lda_within_subject(")]

body = '''def _class_patterns(Z, y, n_cls, sub=6):
    """Per-subject class-mean spatiotemporal patterns, (d, n_cls * T_sub)."""
    P = [Z[y == c].mean(0)[:, ::sub] for c in range(n_cls)]
    return np.concatenate(P, axis=1)


def fit_alignment(Zs, ys, n_cls, n_iter=6):
    """Iterative Procrustes onto a group template.

    Each subject gets an orthogonal W_i mapping its own space into one common space, so the
    big model downstream sees aligned signals. Subject differences here are rotations, not
    offsets -- centring, scaling, indicators and scatter correction all failed to remove them
    -- and a rotation has to be undone at the front, not after the representation is built.

    LEAKAGE GUARD: patterns and template come from training trials only. Test trials are never
    used to fit W_i; they are only transformed by it.
    """
    pats = [_class_patterns(Z, y, n_cls) for Z, y in zip(Zs, ys)]
    tmpl = np.mean(pats, axis=0)
    W = [np.eye(p.shape[0]) for p in pats]
    for _ in range(n_iter):
        for i, p in enumerate(pats):
            U, _, Vt = np.linalg.svd(p @ tmpl.T)
            W[i] = (U @ Vt).T
        tmpl = np.mean([W[i] @ p for i, p in enumerate(pats)], axis=0)
    return W


def _lda_within_subject(Ftr, ytr, str_, Fte, n_cls):
    d = Ftr.shape[1]
    Sw = np.zeros((d, d)); n_cell = 0
    for s_ in np.unique(str_):
        for c in range(n_cls):
            m = (str_ == s_) & (ytr == c)
            if m.sum() < 3: continue
            Z = Ftr[m] - Ftr[m].mean(0)
            Sw += Z.T @ Z; n_cell += m.sum() - 1
    Sw /= max(n_cell, 1)
    lam = 0.3
    Sw = (1 - lam) * Sw + lam * np.trace(Sw) / d * np.eye(d)
    mu = np.stack([Ftr[ytr == c].mean(0) for c in range(n_cls)])
    Wc = np.linalg.solve(Sw + 1e-8 * np.eye(d), mu.T).T
    b = -0.5 * np.einsum("cd,cd->c", mu, Wc)
    return np.argmax(Fte @ Wc.T + b, axis=1)


def _big_model(Zal, y, sid, tr, te, K, tau, n_cls):
    """The big model, unchanged: unified delay-augmented SPD manifold + window features,
    pooled across subjects, one shared classifier."""
    C = unified_cov(Zal, 2, tau)
    Cr = recentre(C, sid, tr)
    ts = TangentSpace(metric="logeuclid").fit(Cr[tr])
    F = ts.transform(Cr)
    ed = np.linspace(0, Zal.shape[2], K + 1).astype(int)
    tw, mw = [], []
    for i in range(K):
        seg = Zal[:, :, ed[i]:ed[i + 1]]
        Cw = Covariances("oas").fit_transform(seg)
        t2 = TangentSpace(metric="logeuclid").fit(Cw[tr])
        tw.append(t2.transform(Cw)); mw.append(seg.mean(2))
    G = np.hstack([x.reshape(len(x), -1) for x in tw + mw])
    sc1 = StandardScaler().fit(G[tr]); sc2 = StandardScaler().fit(F[tr])
    A = np.hstack([sc1.transform(G), sc2.transform(F)])
    return clf(A[tr], y[tr], A[te], y[te])


def _one_fold(args):
    (X, y, sid, tr, te, K, tau, n_sub, sd, fi) = args
    n_cls = len(np.unique(y))
    pc = PCA(20).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)
    out = {}
    out["pooled_noalign"] = _big_model(Z, y, sid, tr, te, K, tau, n_cls)

    Zs = [Z[tr][sid[tr] == s_] for s_ in range(n_sub)]
    ys = [y[tr][sid[tr] == s_] for s_ in range(n_sub)]
    Wl = fit_alignment(Zs, ys, n_cls)
    Zal = Z.copy()
    for s_ in range(n_sub):
        m = sid == s_
        Zal[m] = np.einsum("ij,njt->nit", Wl[s_], Z[m])
    out["pooled_ALIGN"] = _big_model(Zal, y, sid, tr, te, K, tau, n_cls)

    Zs2 = [Z[sid == s_] for s_ in range(n_sub)]
    ys2 = [y[sid == s_] for s_ in range(n_sub)]
    Wl2 = fit_alignment(Zs2, ys2, n_cls)
    Zlk = Z.copy()
    for s_ in range(n_sub):
        m = sid == s_
        Zlk[m] = np.einsum("ij,njt->nit", Wl2[s_], Z[m])
    out["align_LEAKY_check"] = _big_model(Zlk, y, sid, tr, te, K, tau, n_cls)

    per = []
    for s_ in range(n_sub):
        mtr, mte = sid[tr] == s_, sid[te] == s_
        if mte.sum() < 11 or mtr.sum() < 33: continue
        idx = np.concatenate([tr[mtr], te[mte]])
        rel_tr = np.arange(mtr.sum()); rel_te = np.arange(mtr.sum(), len(idx))
        per.append(_big_model(Z[idx], y[idx], np.zeros(len(idx), int),
                              rel_tr, rel_te, K, tau, n_cls))
    out["per_subject_samefolds"] = float(np.mean(per))
    return sd, fi, out


'''
s = head + body + s[s.index("def main("):]
s = s.replace('''    NAMES = ["per_subject_samefolds", "pooled", "+subj_center", "+subj_zscore",
             "+subj_center+onehot", "+subj_center+withinLDA"]''',
'''    NAMES = ["per_subject_samefolds", "pooled_noalign", "pooled_ALIGN", "align_LEAKY_check"]''')
open("pooled_align.py", "w").write(s)
print("written pooled_align.py")
