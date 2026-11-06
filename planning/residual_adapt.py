"""Sandwich-Residuals: test-time residual adaptation of a frozen JEPA world model."""

import logging

import torch
import torch.nn as nn

from .adajepa import AdaJEPATrainer

log = logging.getLogger(__name__)

# planner.adapt options forwarded to ResidualTrainer
RESIDUAL_OPTIONS = (
    "streams", "encoder", "output", "encoder_hidden", "output_hidden", "init_scale",
    "linear_lr", "mlp_lr",
)
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def _mlp(d_in, hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


def _zero_(layer):
    nn.init.zeros_(layer.bias)
    nn.init.zeros_(layer.weight)


def _init_mlp_head_(layer, scale):
    """Small random init: a zero output layer would block gradients to the hidden layer."""
    nn.init.zeros_(layer.bias)
    if scale > 0:
        nn.init.normal_(layer.weight, std=scale)
    else:
        nn.init.zeros_(layer.weight)


class AgentResidual(nn.Module):
    """[E_a(a); E_p(p)] -> correction to the action embedding."""

    def __init__(self, action_dim, proprio_dim):
        super().__init__()
        self.net = nn.Linear(action_dim + proprio_dim, action_dim)
        _zero_(self.net)

    def forward(self, act, proprio):
        return self.net(torch.cat([act, proprio], dim=-1))


class VisualResidual(nn.Module):
    """E_v(x) -> correction to the visual embedding."""

    def __init__(self, dim, hidden, init_scale):
        super().__init__()
        self.net = _mlp(dim, hidden, dim)
        _init_mlp_head_(self.net[-1], init_scale)

    def forward(self, visual):
        return self.net(visual)


class OutputResidual(nn.Module):
    """Corrections to the predicted visual and proprio embeddings."""

    def __init__(self, visual_dim, proprio_dim, hidden, init_scale):
        super().__init__()
        self.visual_net = _mlp(visual_dim, hidden, visual_dim)
        self.proprio_net = nn.Linear(proprio_dim, proprio_dim)
        _init_mlp_head_(self.visual_net[-1], init_scale)
        _zero_(self.proprio_net)

    def forward(self, pred_visual, pred_proprio):
        return self.visual_net(pred_visual), self.proprio_net(pred_proprio)


class _StreamSlot:
    """Read/write the visual, proprio and action slots of a packed latent z."""

    def __init__(self, wm):
        if wm.concat_dim != 1:
            raise ValueError(f"residual adaptation requires concat_dim=1, got {wm.concat_dim}")
        self.A, self.P = wm.action_dim, wm.proprio_dim
        self.a_rep, self.p_rep = wm.num_action_repeat, wm.num_proprio_repeat
        self.action_dim, self.proprio_dim = self.A // self.a_rep, self.P // self.p_rep

    def read(self, z):
        """-> (act, proprio, visual): act/proprio (b,t,base) from patch 0, visual (b,t,p,V)."""
        A, P = self.A, self.P
        return (z[:, :, 0, -A:][..., : self.action_dim],
                z[:, :, 0, -(P + A):-A][..., : self.proprio_dim],
                z[..., : -(P + A)])

    def write(self, z, act=None, proprio=None, visual=None):
        """Copy of z with the given slots replaced; act/proprio are tiled to every patch."""
        z = z.clone()
        A, P, n = self.A, self.P, z.shape[2]
        tile = lambda x, rep: x.repeat(1, 1, rep).unsqueeze(2).expand(-1, -1, n, -1)
        if visual is not None:
            z[..., : -(P + A)] = visual
        if proprio is not None:
            z[..., -(P + A):-A] = tile(proprio, self.p_rep)
        if act is not None:
            z[..., -A:] = tile(act, self.a_rep)
        return z


def attach_residual(wm, streams=("action",), encoder="mlp", output="mlp", encoder_hidden=64,
                    output_hidden=64, init_scale=1e-4):
    """Attach the residuals by patching wm.predict / wm.encode_obs."""
    if list(streams) not in ([], ["action"]):
        raise ValueError(f"streams must be [] or ['action'], got {list(streams)!r}")
    for name, kind in (("encoder", encoder), ("output", output)):
        if kind not in ("none", "mlp"):
            raise ValueError(f"{name} residual must be 'none' or 'mlp', got {kind!r}")
    slot, V = _StreamSlot(wm), wm.encoder.emb_dim
    rng = torch.get_rng_state()  # keep init off the global RNG stream
    mods = nn.ModuleDict()
    if streams:
        mods["agent"] = AgentResidual(slot.action_dim, slot.proprio_dim)
    if encoder == "mlp":
        mods["encoder"] = VisualResidual(V, encoder_hidden, init_scale)
    if output == "mlp":
        mods["output"] = OutputResidual(V, slot.proprio_dim, output_hidden, init_scale)
    torch.set_rng_state(rng)
    if not mods:
        raise ValueError("residual adaptation needs at least one of streams / encoder / output")
    mods.to(next(wm.parameters()).device)

    orig_predict, orig_encode_obs = wm.predict, wm.encode_obs

    def encode_obs(obs):
        z = dict(orig_encode_obs(obs))
        z["visual"] = z["visual"] + mods["encoder"](z["visual"])
        return z

    def predict(z):
        # act and proprio are read from patch 0 and written back to every patch
        act, prop, _ = slot.read(z)
        if "agent" in mods:
            act = act + mods["agent"](act, prop)
        pred = orig_predict(slot.write(z, act=act, proprio=prop))
        if "output" in mods:
            _, pred_prop, pred_vis = slot.read(pred)
            d_vis, d_prop = mods["output"](pred_vis, pred_prop)
            pred = slot.write(pred, proprio=pred_prop + d_prop, visual=pred_vis + d_vis)
        return pred

    wm.predict = predict
    if "encoder" in mods:
        wm.encode_obs = encode_obs
    return mods


class ResidualTrainer(AdaJEPATrainer):
    """AdaJEPATrainer whose only adapted parameters are the attached residuals."""

    def __init__(self, wm, lr, steps=1, optimizer_name="adam", linear_lr=None, mlp_lr=None,
                 **attach_kwargs):
        self.residual = attach_residual(wm, **attach_kwargs)
        self.kind_lr = {"linear": linear_lr, "mlp": mlp_lr}
        super().__init__(wm=wm, lr=lr, steps=steps, optimizer_name=optimizer_name,
                         finetune_encoder=False)
        for name, _, lr_i, n in self._parameter_groups():
            log.info("Residual '%s': %d params, lr=%.3e", name, n, lr_i)

    def _select_predictor_params(self):
        return list(self.residual.parameters())

    @property
    def _encode_needs_grad(self):
        """The encoder residual needs gradients through encoding."""
        return "encoder" in self.residual

    def _parameter_groups(self):
        """(name, params, lr, n_params) per group: linear_lr for linear residuals, mlp_lr for MLPs."""
        groups = []
        for name, mod in self.residual.items():
            if isinstance(mod, OutputResidual):
                parts = [("output_visual", mod.visual_net, "mlp"),
                         ("output_proprio", mod.proprio_net, "linear")]
            elif isinstance(mod, VisualResidual):
                parts = [(name, mod, "mlp")]
            else:
                parts = [(name, mod, "linear")]
            for gname, gmod, kind in parts:
                params = list(gmod.parameters())
                lr = self.kind_lr[kind]
                if lr is None:
                    lr = self.lr
                groups.append((gname, params, lr, sum(p.numel() for p in params)))
        return groups

    def _make_optimizer(self):
        if self.optimizer_name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer_name!r}")
        return OPTIMIZERS[self.optimizer_name](
            [{"params": p, "lr": lr} for _, p, lr, _ in self._parameter_groups()])

    @property
    def param_distance_by_tensor(self):
        """||delta|| per residual tensor since the last reset."""
        saved = {id(t): s for t, s in self._snapshot}
        with torch.no_grad():
            return {n: float((p - saved[id(p)]).float().norm())
                    for n, p in self.residual.named_parameters() if id(p) in saved}

    def _set_train_mode(self, mode):
        """No-op: the frozen predictor stays in eval mode."""
