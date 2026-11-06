"""Cache DINOv2-S patch features (fp16) of every frame next to the h5 file."""

import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from stable_worldmodel.wm.prejepa.module import create_backbone

H5 = Path(sys.argv[1])
OUT = H5.with_suffix(".dinov2_small.npy")
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
B = 500

enc = create_backbone("dinov2_small").cuda().eval().requires_grad_(False)
with h5py.File(H5, "r") as f:
    px = f["pixels"]
    N = px.shape[0]
    feats = np.lib.format.open_memmap(OUT, "w+", np.float16, (N, 256, enc.config.hidden_size))
    for s in range(0, N, B):
        x = torch.from_numpy(px[s:s + B]).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            h = enc((x - MEAN) / STD, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        feats[s:s + B] = h.half().cpu().numpy()
        if s % 50000 == 0:
            print(s, N, flush=True)
    feats.flush()
print("wrote", OUT, feats.shape)
