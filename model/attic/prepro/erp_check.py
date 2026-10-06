import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd

SUBS = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
VOWELS = ["a", "e", "i", "o", "u"]
STOPS = ["b", "p", "d", "t"]

def load(sub):
    d = np.load("npz/sub-%s_ses-02.npz" % sub, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def vowels_for(sub):
    ev = pd.read_csv("events/sub-%s_phonemes_events.tsv" % sub, sep="\t")
    st = ev[ev.trial_type == "stimulus"].reset_index(drop=True)
    return st.phoneme2.astype(str).str.strip().values

def baseline(X, t):
    bm = (t >= -0.2) & (t <= -0.11)
    return X - X[:, :, bm].mean(2, keepdims=True)

def evoked_stats(Xb, t, labels=None):
    pre = (t >= -0.2) & (t <= -0.11)
    post = (t >= 0.05) & (t <= 0.35)
    ga = Xb.mean(0)
    gfp = ga.std(0)
    snr = gfp[post].mean() / (gfp[pre].mean() + 1e-12)
    out = {"gfp_post": gfp[post].mean(), "gfp_pre": gfp[pre].mean(), "evoked_snr": snr}
    if labels is not None:
        means = np.stack([Xb[labels == u].mean(0) for u in np.unique(labels)], 0)
        between = means.var(0).mean()
        within = np.mean([Xb[labels == u].var(0).mean() / max(1, (labels == u).sum())
                          for u in np.unique(labels)])
        out["between_over_within"] = between / (within + 1e-20)
    return out

rows = []
print("ERP DIAGNOSTIC: is there an auditory evoked response in each task?\n")
print("%-5s | %-28s | %-28s" % ("sub", "single-phoneme (control)", "phonemes CV (control)"))
print("%-5s | %8s %8s %9s | %8s %8s %9s" % ("", "GFPpost", "SNR", "B/W(ph)", "GFPpost", "SNR", "B/W(vow)"))
for sub in SUBS:
    X, t, meta = load(sub)
    ph = meta.phoneme.values; task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_iso = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    Xi = baseline(X[m_iso], t)
    si = evoked_stats(Xi, t, ph[m_iso])
    m_all = task == "phonemes"
    try:
        vw = vowels_for(sub)
    except Exception:
        continue
    if m_all.sum() != len(vw):
        continue
    m_syl = m_all & ctrl
    vsel = vw[ctrl[m_all]]
    Xs = baseline(X[m_syl], t)
    ss = evoked_stats(Xs, t, vsel)
    rows.append((si, ss))
    print("%-5s | %8.3f %8.2f %9.4f | %8.3f %8.2f %9.4f" % (
        sub, si["gfp_post"], si["evoked_snr"], si["between_over_within"],
        ss["gfp_post"], ss["evoked_snr"], ss["between_over_within"]), flush=True)

A = np.array([[r[0]["gfp_post"], r[0]["evoked_snr"], r[0]["between_over_within"],
               r[1]["gfp_post"], r[1]["evoked_snr"], r[1]["between_over_within"]] for r in rows])
print("\n=== GROUP MEANS ===")
print("  single-phoneme : GFP_post %.3f  evoked-SNR %.2f  between/within %.4f" % (A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean()))
print("  phonemes CV    : GFP_post %.3f  evoked-SNR %.2f  between/within %.4f" % (A[:, 3].mean(), A[:, 4].mean(), A[:, 5].mean()))
print("\n  evoked-SNR ratio (iso/syl): %.2f" % (A[:, 1].mean() / A[:, 4].mean()))
print("  between/within ratio (iso/syl): %.2f" % (A[:, 2].mean() / A[:, 5].mean()))
print("\n  interpretation: evoked-SNR ~1 means NO evoked response;")
print("                  between/within is the class-separability of the mean responses.")
