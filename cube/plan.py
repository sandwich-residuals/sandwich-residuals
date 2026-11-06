"""OGBench-Cube: test-time adaptation of a DINO-WM model under the AdaJEPA protocol."""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import io
import json
import logging
import time
from contextlib import redirect_stdout
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401  (compression filter used by `pixels`)
import hydra
import mujoco
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.wm.prejepa.module import create_backbone
from torchvision.transforms import v2 as transforms

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "checkpoints" / "cube_dinowm" / "weights_epoch_10.pt"
H5 = REPO / "data" / "cube_single_expert.h5"
log = logging.getLogger("cube")

# 5-block horizon of 5 env steps, goal 25 steps ahead, 50-step budget, 3-frame history
HORIZON, BLOCK, GOAL_OFFSET, BUDGET, HIST = 5, 5, 25, 50, 3
CEM = dict(num_samples=200, var_scale=1.0, n_steps=10, topk=30)
BUFFER = 5  # adaptation buffer: the 5 most recent transitions

SHIFTS = {
    "default": {},
    "light_0.3": {"variation_values": {"light.intensity": [0.3]}},
    "gain_0.5": {"arm_gain": 0.5},
    "camera_10": {"variation_values": {"camera.angle_delta": [[10.0, 10.0]]}},
}

LAYER_ARMS = {  # AdaJEPA: adapted pretrained modules and their learning rates
    "frozen": [],
    "predlast": [("predictor.transformer.layers.5", 5e-4), ("predictor.transformer.norm", 5e-4)],
    "predfirst": [("predictor.transformer.layers.0", 5e-4)],
}


def load_model():
    cfg = OmegaConf.create(json.loads((CKPT.parent / "config.json").read_text()))
    model = hydra.utils.instantiate(cfg, encoder=create_backbone(cfg.encoder.name))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    return model.cuda().eval().requires_grad_(False)


class Residual(nn.Module):
    """base(x) + r(base(x)) on the first d feature dims (all if d is None)."""

    def __init__(self, base, res, d=None):
        super().__init__()
        self.base, self.res, self.d = base, res, d

    def forward(self, *a, **k):
        y = self.base(*a, **k)
        if self.d is None:
            return y + self.res(y)
        return torch.cat([y[..., :self.d] + self.res(y[..., :self.d]), y[..., self.d:]], -1)


def _mlp(d, h, init_scale=1e-4):
    m = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    nn.init.normal_(m[-1].weight, std=init_scale)
    nn.init.zeros_(m[-1].bias)
    return m


def _zero(m):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


def attach(model, method):
    """Return [(name, params, lr)] of the adapted parameters; "ours" attaches the residuals."""
    if method in LAYER_ARMS:
        mods = dict(model.named_modules())
        return [(n, list(mods[n].parameters()), lr) for n, lr in LAYER_ARMS[method]]
    d, ae = model.backbone.config.hidden_size, model.extra_encoders["action"]
    act = Residual(ae, _zero(nn.Linear(ae.emb_dim, ae.emb_dim)))
    act.emb_dim = ae.emb_dim
    model.extra_encoders["action"] = act
    model.residuals = nn.ModuleDict({"encoder": Residual(model._encode_image, _mlp(d, 64)),
                                     "output": Residual(model.predict, _mlp(d, 64), d)})
    model._encode_image, model.predict = model.residuals["encoder"].__call__, model.residuals["output"].__call__
    model.cuda()
    return [("action", list(act.res.parameters()), 1e-2),
            ("encoder", list(model.residuals["encoder"].res.parameters()), 5e-4),
            ("output", list(model.residuals["output"].res.parameters()), 5e-4)]


def load_data(n, seed):
    """Start states, goals and preceding history from the dataset (LeWorldModel protocol)."""
    f = h5py.File(H5, "r")
    act = f["action"][:]
    scaler = StandardScaler().fit(act[~np.isnan(act).any(1)])
    off, ln, pos = f["ep_offset"][:], f["ep_len"][:], f["privileged_block_0_pos"]
    rs, out, h = np.random.RandomState(seed), [], (HIST - 1) * BLOCK
    for _ in range(n):
        e = rs.randint(len(ln))
        i = off[e] + rs.randint(h, ln[e] - GOAL_OFFSET)
        g = i + GOAL_OFFSET
        out.append(dict(qpos=f["qpos"][i], qvel=f["qvel"][i], goal=f["pixels"][g],
                        goal_pos=pos[g], goal_quat=f["privileged_block_0_quat"][g],
                        hist_pixels=f["pixels"][i - h:i:BLOCK],
                        hist_action=scaler.transform(act[i - h:i]).reshape(HIST - 1, -1)))
    return out, scaler


class Env:
    def __init__(self, shift):
        self.env = gym.make("swm/OGBCube-v0", env_type="single", ob_type="pixels", render_mode="rgb_array")
        self.u, self.opts = self.env.unwrapped, dict(shift)
        self.gain = self.opts.pop("arm_gain", 1.0)
        self.env.reset(seed=0)
        m, b = self.u._model, self.u._model.body("object_0").id
        self.base = m.body_mass[b].copy(), m.body_inertia[b].copy(), m.actuator_gainprm[:6, 0].copy(), m.actuator_biasprm[:6, 1].copy()

    def reset(self, t, seed):
        self.env.reset(seed=seed, options=self.opts)  # visual shifts use the env's variation space
        m, d = self.u._model, self.u._data
        b = m.body("object_0").id
        # reset() may recompile the model: set the nominal cube mass and the (scaled) arm-joint kp
        m.body_mass[b], m.body_inertia[b] = self.base[0], self.base[1]
        m.actuator_gainprm[:6, 0], m.actuator_biasprm[:6, 1] = self.base[2] * self.gain, self.base[3] * self.gain
        mujoco.mj_setConst(m, d)
        self.u.set_state(t["qpos"].astype(np.float64), t["qvel"].astype(np.float64))
        self.u.set_target_pos(0, t["goal_pos"], t["goal_quat"])
        mujoco.mj_forward(m, d)
        return self.u.render()

    def step_block(self, actions):
        """Execute (BLOCK, 5) env-space actions; return the frame after the block and success."""
        ok = False
        for a in actions:
            obs, _, _, _, info = self.env.step(np.clip(a, -1, 1).astype(np.float32))
            ok |= bool(info["success"])
        return obs, ok

    def dist(self):
        d = self.u._data
        return float(np.linalg.norm(d.joint("object_joint_0").qpos[:3] - d.mocap_pos[self.u._cube_target_mocap_ids[0]]))


class Cost:
    """get_cost with the executed history blocks prepended to every candidate."""

    def __init__(self, model):
        self.model, self.hist = model, None

    def get_cost(self, info, cand):
        return self.model.get_cost(info, torch.cat([self.hist.expand(*cand.shape[:2], -1, -1), cand], 2))


class Planner:
    def __init__(self, model, groups, seed):
        self.m, self.params = model, [p for _, ps, _ in groups for p in ps]
        self.snap = [p.detach().clone() for p in self.params]
        self.opt = torch.optim.Adam([{"params": ps, "lr": lr} for _, ps, lr in groups]) if groups else None
        self.cost = Cost(model)
        self.solver = swm.solver.CEMSolver(self.cost, device="cuda", seed=seed, **CEM)
        self.solver.configure(action_space=gym.spaces.Box(-1, 1, (1, 5)), n_envs=1,
                              config=swm.PlanConfig(horizon=HORIZON, receding_horizon=1, action_block=BLOCK))
        self.tf = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ])
        self.ctx, self.d, self.n = model.history_size, model.backbone.config.hidden_size, 0
        for n, ps, lr in groups:
            log.info("adapt '%s': %d params, lr=%.0e", n, sum(p.numel() for p in ps), lr)

    @torch.no_grad()
    def reset(self):
        for p, s in zip(self.params, self.snap):
            p.copy_(s)
        if self.opt:
            self.opt.state.clear()

    def _img(self, frames):
        return torch.stack([self.tf(torch.from_numpy(np.asarray(f)).permute(2, 0, 1)) for f in frames])

    def info(self, frames, goal):
        self.n += 1  # PreJEPA caches encodings per (id, step_idx); a fresh key per replan
        return {"pixels": self._img(frames)[None], "goal": self._img([goal])[None], "action": torch.zeros(1, 1, 5 * BLOCK),
                "id": torch.zeros(1, 1, dtype=torch.long), "step_idx": torch.full((1, 1), self.n)}

    def plan(self, frames, hist, goal, init):
        """CEM from HIST real frames; returns (1, HORIZON, 25) standardised actions."""
        with torch.autocast("cuda", torch.float16):
            self.cost.hist = hist
            with redirect_stdout(io.StringIO()):  # the solver prints its solve time
                return self.solver.solve(self.info(frames, goal), init_action=init)["actions"]

    def loss(self, buf):
        """Sliding-window next-embedding MSE on the buffer, stop-gradient targets (AdaJEPA)."""
        frames = [buf[0][0]] + [b[2] for b in buf]
        z = self.m._encode_image(self._img(frames)[None].cuda())[0]
        ae = self.m.extra_encoders["action"](torch.stack([b[1] for b in buf])[None].cuda())[0]
        e = torch.cat([z[:-1], ae[:, None].expand(-1, z.shape[1], -1)], -1)
        w = min(self.ctx, len(buf)); n = len(buf) - w + 1
        return sum(F.mse_loss(self.m.predict(e[t:t + w][None])[0][..., :self.d], z[t + 1:t + 1 + w].detach()) for t in range(n)) / n

    def adapt(self, buf):
        if not self.opt:
            with torch.no_grad():
                return self.loss(buf).item()
        self.opt.zero_grad()
        loss = self.loss(buf)
        loss.backward()
        self.opt.step()
        return loss.item()

    def param_dist(self):
        with torch.no_grad():
            return float(sum(((p - s) ** 2).sum() for p, s in zip(self.params, self.snap)).sqrt()) if self.params else 0.0


def run_episode(planner, env, scaler, t, ep, seed, records):
    planner.reset()
    frames = list(t["hist_pixels"]) + [env.reset(t, seed)]
    hist = torch.as_tensor(t["hist_action"], dtype=torch.float32, device="cuda")[None, None]  # (1, 1, HIST-1, 25)
    init, buf, ok = None, [], False
    for k in range(BUDGET // BLOCK):
        t0 = time.time()
        plan = planner.plan(frames[-HIST:], hist, t["goal"], init)
        a = plan[0, 0].clone()  # leave the solver's inference mode
        nxt, ok = env.step_block(scaler.inverse_transform(a.numpy().reshape(BLOCK, -1)))
        buf = (buf + [(frames[-1], a, nxt)])[-BUFFER:]
        frames.append(nxt)
        rec = {"sample_idx": ep, "step": k + 1, "success": ok, "t_plan_s": time.time() - t0, "adapt/param_dist": planner.param_dist()}
        rec["adajepa/pred_loss"] = planner.adapt(buf)
        records.append(rec)
        if ok:
            break
        hist = torch.cat([hist[:, :, 1:], a.cuda().view(1, 1, 1, -1)], 2)
        init = plan[:, 1:]  # warm start from the unexecuted part of the plan
    return ok, env.dist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=[*LAYER_ARMS, "ours"])
    ap.add_argument("--condition", default="default", choices=list(SHIFTS))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--n_evals", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for h in (logging.FileHandler(args.out / "run.log"), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s")); log.addHandler(h)
    log.setLevel(logging.INFO); log.propagate = False  # stable_pretraining reroutes the root logger
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = load_model()
    groups = attach(model, args.method)
    for _, ps, _ in groups:
        for p in ps:
            p.requires_grad_(True)
    planner = Planner(model, groups, args.seed)
    env = Env(SHIFTS[args.condition])
    targets, scaler = load_data(args.n_evals, args.seed)

    records, oks, dists, t0 = [], [], [], time.time()
    for i, t in enumerate(targets):
        ok, dist = run_episode(planner, env, scaler, t, i, args.seed + i, records)
        oks.append(ok); dists.append(dist)
        log.info("episode %d/%d  success=%s  dist=%.3f  running rate=%.2f", i + 1, len(targets), ok, dist, np.mean(oks))
    n_params = sum(p.numel() for p in planner.params)
    with open(args.out / "per_iter_logs.json", "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in records))
    with open(args.out / "logs.json", "w") as f:
        f.write(json.dumps({"final_eval/success_rate": float(np.mean(oks)), "final_eval/mean_state_dist": float(np.mean(dists)),
                            "n_evals": len(targets), "method": args.method, "condition": args.condition, "seed": args.seed,
                            "adapted_params": n_params, "wall_s": time.time() - t0}) + "\n")
    log.info("FINAL success_rate=%.2f  mean_dist=%.3f  adapted_params=%d  wall=%.0fs",
             np.mean(oks), np.mean(dists), n_params, time.time() - t0)


if __name__ == "__main__":
    main()
