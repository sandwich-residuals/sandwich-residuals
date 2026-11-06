"""Write the first N readable episodes of cube_single_expert.h5 to a new h5."""

import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

SRC = Path(__file__).resolve().parents[2] / "data" / "cube_single_expert.h5"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DST = SRC.with_name(f"cube_single_expert_{N}.h5")

with h5py.File(SRC, "r") as f, h5py.File(DST, "w") as g:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    keep = []
    for e in range(len(ln)):
        if len(keep) == N:
            break
        try:
            f["pixels"][off[e]:off[e] + ln[e]]
            keep.append(e)
        except OSError:
            print("skip corrupt episode", e, flush=True)
    rows = np.concatenate([np.arange(off[e], off[e] + ln[e]) for e in keep])
    new_len = ln[keep]
    g["ep_len"], g["ep_offset"] = new_len, np.concatenate([[0], np.cumsum(new_len)[:-1]])
    g["ep_idx"] = np.repeat(np.arange(N), new_len).astype(f["ep_idx"].dtype)
    for k in f:
        if k in ("ep_len", "ep_offset", "ep_idx"):
            continue
        d = f[k]
        if k == "pixels":
            out = g.create_dataset(k, (len(rows), *d.shape[1:]), d.dtype, chunks=d.chunks,
                                   **hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
            for i, e in enumerate(keep):
                s = int(g["ep_offset"][i])
                out[s:s + ln[e]] = d[off[e]:off[e] + ln[e]]
                if i % 200 == 0:
                    print(f"pixels {i}/{N}", flush=True)
        else:
            g.create_dataset(k, data=d[:][rows], dtype=d.dtype)
print("wrote", DST, "episodes", len(keep), "rows", len(rows))
