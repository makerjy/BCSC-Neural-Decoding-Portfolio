s = open("pooled_unified.py").read()

old = s[s.index("def _one_fold("):s.index("def main(")]
new = '''def _lda_within_subject(Ftr, ytr, str_, Fte, n_cls):
    """LDA whose within-class scatter is pooled WITHIN (subject, class).

    Standard LDA pools within-class scatter across subjects, so subject variance inflates it
    and the discriminant directions bend toward whatever separates people. Estimating scatter
    inside each (subject, class) cell removes subject variance from the denominator, which is
    the discriminant-level version of factorising subject out of phoneme.
    """
    d = Ftr.shape[1]
    Sw = np.zeros((d, d))
    n_cell = 0
    for s_ in np.unique(str_):
        for c in range(n_cls):
            m = (str_ == s_) & (ytr == c)
            if m.sum() < 3: continue
            Z = Ftr[m] - Ftr[m].mean(0)
            Sw += Z.T @ Z
            n_cell += m.sum() - 1
    Sw /= max(n_cell, 1)
    lam = 0.3
    Sw = (1 - lam) * Sw + lam * np.trace(Sw) / d * np.eye(d)
    mu = np.stack([Ftr[ytr == c].mean(0) for c in range(n_cls)])
    Wc = np.linalg.solve(Sw + 1e-8 * np.eye(d), mu.T).T
    b = -0.5 * np.einsum("cd,cd->c", mu, Wc)
    return np.argmax(Fte @ Wc.T + b, axis=1)


def _subject_center(F, sid, tr, scale=False):
    out = F.copy()
    for s_ in np.unique(sid):
        m = sid == s_
        mtr = m[tr] if len(tr) == len(sid) else (sid[tr] == s_)
        ref = F[tr][mtr]
        if len(ref) < 5: continue
        out[m] = F[m] - ref.mean(0)
        if scale:
            out[m] = out[m] / (ref.std(0) + 1e-8)
    return out


def _one_fold(args):
    (X, y, sid, tr, te, K, tau, n_sub, sd, fi) = args
    n_cls = len(np.unique(y))
    pc = PCA(20).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
    A = np.stack([pc.transform(z.T).T for z in X], 0)
    C20 = unified_cov(A, 2, tau)
    Cr = recentre(C20, sid, tr)
    ts = TangentSpace(metric="logeuclid").fit(Cr[tr])
    Fr = ts.transform(Cr)
    wa, wb, mua, mub = window_parts(X[tr], X[te], K, 20)
    a, b = stack([wa, mua, Fr[tr]], [wb, mub, Fr[te]])
    full = np.zeros((len(X), a.shape[1]))
    full[tr], full[te] = a, b

    out = {}
    out["pooled"] = clf(a, y[tr], b, y[te])
    Fc = _subject_center(full, sid, tr)
    out["+subj_center"] = clf(Fc[tr], y[tr], Fc[te], y[te])
    Fz = _subject_center(full, sid, tr, scale=True)
    out["+subj_zscore"] = clf(Fz[tr], y[tr], Fz[te], y[te])
    oh = np.eye(n_sub)[sid]
    a2 = np.hstack([Fc[tr], oh[tr]]); b2 = np.hstack([Fc[te], oh[te]])
    out["+subj_center+onehot"] = clf(a2, y[tr], b2, y[te])
    pred = _lda_within_subject(Fc[tr], y[tr], sid[tr], Fc[te], n_cls)
    out["+subj_center+withinLDA"] = balanced_accuracy_score(y[te], pred)

    per = []
    for s_ in range(n_sub):
        mtr, mte = sid[tr] == s_, sid[te] == s_
        if mte.sum() < 11 or mtr.sum() < 33: continue
        Xs, Xq = X[tr][mtr], X[te][mte]
        ws, wq, ms, mq = window_parts(Xs, Xq, K, 20)
        pcs = PCA(20).fit(Xs.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        As = np.stack([pcs.transform(z.T).T for z in Xs], 0)
        Aq = np.stack([pcs.transform(z.T).T for z in Xq], 0)
        Cs, Cq = unified_cov(As, 2, tau), unified_cov(Aq, 2, tau)
        t2 = TangentSpace(metric="logeuclid").fit(Cs)
        aa, bb = stack([ws, ms, t2.transform(Cs)], [wq, mq, t2.transform(Cq)])
        per.append(clf(aa, y[tr][mtr], bb, y[te][mte]))
    out["per_subject_samefolds"] = float(np.mean(per))
    return sd, fi, out


'''
s = s.replace(old, new)
s = s.replace('''    NAMES = ["per_subject", "pooled_naive", "pooled_rc", "pooled_rc_rich",
             "pooled_rc_rich+win", "pooled_rc_persubj_clf"]''',
'''    NAMES = ["per_subject_samefolds", "pooled", "+subj_center", "+subj_zscore",
             "+subj_center+onehot", "+subj_center+withinLDA"]''')
s = s.replace('    base = np.array(R["per_subject"])', '    base = np.array(R["per_subject_samefolds"])')
s = s.replace('from sklearn.metrics import balanced_accuracy_score',
              'from sklearn.metrics import balanced_accuracy_score')
open("pooled_factor.py", "w").write(s)
print("written pooled_factor.py")
