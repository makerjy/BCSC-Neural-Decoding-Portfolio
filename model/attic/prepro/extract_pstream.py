import argparse
import glob
import os
import numpy as np
import pandas as pd

P_STD = 0.3


def main(data_dir: str, out: str) -> None:
    files = sorted(glob.glob(os.path.join(data_dir, "sub-S*_ses-02.npz")))
    files = [f for f in files if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    if not files:
        raise SystemExit("no npz found in %s" % data_dir)
    segs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        X, t = d["X"].astype(float), d["times"]
        meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target"]})
        k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
        wm = (t >= 0.0) & (t <= 0.5)
        Xk = X[k][:, :, wm]
        flat = Xk.transpose(0, 2, 1).reshape(-1, Xk.shape[1])
        flat = flat - flat.mean(0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        segs.append(np.einsum("c,nct->nt", Vt[0], Xk).reshape(-1))
        print("  %-10s %d samples" % (os.path.basename(f).split("_")[0], segs[-1].size))
    stream = np.concatenate(segs)
    stream = (stream / (stream.std() + 1e-9) * P_STD).astype(np.float32)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, stream)
    print("\nsaved %s  shape %s  %.1f MB  std %.3f" % (
        out, stream.shape, stream.nbytes / 1e6, stream.std()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="npz")
    ap.add_argument("--out", default="pstream.npy")
    a = ap.parse_args()
    main(a.data, a.out)
