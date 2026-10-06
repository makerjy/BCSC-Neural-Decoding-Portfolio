import os
import subprocess
import numpy as np
import pandas as pd
import collections

SUBS = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
BASE = "https://s3.amazonaws.com/openneuro.org/ds006104"
OUT = "events"
os.makedirs(OUT, exist_ok=True)

for s in SUBS:
    for task in ["phonemes", "single-phoneme"]:
        dst = os.path.join(OUT, "sub-%s_%s_events.tsv" % (s, task))
        if os.path.exists(dst):
            continue
        url = "%s/sub-%s/ses-02/eeg/sub-%s_ses-02_task-%s_events.tsv" % (BASE, s, s, task)
        r = subprocess.run(["curl", "-s", "--max-time", "60", "-o", dst, url], capture_output=True)
        if os.path.getsize(dst) < 200:
            os.remove(dst)
            print("  miss %s %s" % (s, task))
print("downloaded:", len(os.listdir(OUT)))

f = os.path.join(OUT, "sub-S04_phonemes_events.tsv")
df = pd.read_csv(f, sep="\t")
st = df[df.trial_type == "stimulus"]
print("\n=== phonemes task (S04) stimulus events: n=%d ===" % len(st))
print("phoneme1:", dict(collections.Counter(st.phoneme1.astype(str))))
print("phoneme2 (VOWEL CONTEXT):", dict(collections.Counter(st.phoneme2.astype(str))))
print("phoneme3:", dict(collections.Counter(st.phoneme3.astype(str))))
print("\nCV pairs (C x V):")
print(pd.crosstab(st.phoneme1.astype(str), st.phoneme2.astype(str)))

f2 = os.path.join(OUT, "sub-S04_single-phoneme_events.tsv")
if os.path.exists(f2):
    d2 = pd.read_csv(f2, sep="\t")
    s2 = d2[d2.trial_type == "stimulus"]
    print("\n=== single-phoneme (S04): n=%d ===" % len(s2))
    print("phoneme1:", dict(collections.Counter(s2.phoneme1.astype(str))))
    print("phoneme2:", dict(collections.Counter(s2.phoneme2.astype(str))))
