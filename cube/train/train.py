"""Train the DINO-WM cube model on cached DINOv2 features; writes checkpoints/cube_dinowm/."""

import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset

import stable_worldmodel  # noqa: F401
from stable_worldmodel.wm.prejepa.module import create_backbone
from stable_worldmodel.wm.utils import save_pretrained

H5 = Path(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
os.environ["STABLEWM_HOME"] = str(Path(__file__).resolve().parents[2])  # save_pretrained writes to <repo>/checkpoints/

cfg = OmegaConf.load(Path(__file__).with_name("config.yaml"))
FS, H, NP = cfg.frameskip, cfg.wm.history_size, cfg.wm.num_preds
T = H + NP
SPAN = T * FS

f = h5py.File(H5, "r")
act, off, ln = f["action"][:], f["ep_offset"][:], f["ep_len"][:]
ok = act[~np.isnan(act).any(1)]
act = np.nan_to_num((act - ok.mean(0)) / ok.std(0)).astype(np.float32)
feats = np.load(H5.with_suffix(".dinov2_small.npy"), mmap_mode="r")
clips = np.array([o + s for o, l in zip(off, ln) for s in range(l - SPAN + 1)])
perm = torch.randperm(len(clips), generator=torch.Generator().manual_seed(cfg.seed)).numpy()
n_train = int(cfg.train_split * len(clips))


class Clips(Dataset):
    def __init__(self, starts):
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(np.ascontiguousarray(feats[s:s + SPAN:FS])), torch.from_numpy(act[s:s + SPAN].reshape(T, -1))


loader = DataLoader(Clips(clips[perm[:n_train]]), cfg.batch_size, shuffle=True, num_workers=8, drop_last=True,
                    pin_memory=True, persistent_workers=True)
vloader = DataLoader(Clips(clips[perm[n_train:]]), cfg.batch_size, num_workers=8)

enc = create_backbone(cfg.backbone.name)
D = enc.config.hidden_size
with open_dict(cfg):
    cfg.model.predictor.dim = D + cfg.wm.encoding.action
    cfg.model.predictor.num_patches = (cfg.image_size // cfg.patch_size) ** 2
    cfg.model.extra_encoders = {"_target_": "torch.nn.ModuleDict", "modules": {"action": {
        "_target_": "stable_worldmodel.wm.prejepa.module.Embedder", "in_chans": act.shape[1] * FS, "emb_dim": int(cfg.wm.encoding.action)}}}
model = hydra.utils.instantiate(cfg.model, encoder=enc).cuda()
model.backbone.eval().requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, **{k: v for k, v in cfg.optimizer.items() if k != "type"})
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * len(loader))
scaler = torch.amp.GradScaler()
print(f"clips train {n_train} val {len(clips) - n_train}, steps/epoch {len(loader)}, trainable {sum(p.numel() for p in params)}", flush=True)


def loss_fn(feat, a):
    feat, a = feat.cuda(non_blocking=True).float(), a.cuda(non_blocking=True)
    ae = model.extra_encoders["action"](a)
    e = torch.cat([feat, ae[:, :, None].expand(-1, -1, feat.shape[2], -1)], -1)
    pred = model.predict(e[:, :H])
    return F.mse_loss(pred[..., :D], e[:, NP:, :, :D].detach())


for ep in range(1, EPOCHS + 1):
    model.train()
    model.backbone.eval()
    t0, tot = time.time(), 0.0
    for feat, a in loader:
        with torch.autocast("cuda", torch.float16):
            loss = loss_fn(feat, a)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.trainer.gradient_clip_val)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        val = np.mean([loss_fn(x, y).item() for x, y in vloader])
    print(f"epoch {ep} train {tot / len(loader):.5f} val {val:.5f} {time.time() - t0:.0f}s", flush=True)
    save_pretrained(model, cfg.output_model_name, config=cfg.model, filename=f"weights_epoch_{ep}.pt")
