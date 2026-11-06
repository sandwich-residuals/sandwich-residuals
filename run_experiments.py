"""Run the paper's AdaJEPA-benchmark experiments; finished runs are skipped."""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
SEEDS = [1001, 1002, 1003, 1004, 1005]
N_EVALS = 50

TASKS = {
    "mediummaze": ("adajepa_plan_cem_maze.yaml", "mediummaze_dynamics_shift", "point_maze_medium"),
    "diversemaze": ("adajepa_plan_cem_diversemaze.yaml", "diversemaze_layout_shift", "diverse_maze"),
    "pusht": ("adajepa_plan_cem_pushobj.yaml", "pusht_visual_shift", "pushobj_eval/val_T/plan_targets.pkl"),
    "pushshape": ("adajepa_plan_cem_pushobj.yaml", "pushobj_shape_shift", "pushobj_eval/val_T/plan_targets.pkl"),
}

BLUR = {"ood_corruption": "blur", "ood_level": 2}
KV = {"env_kwargs_override.kv_scale": 2}
RED_ANCHOR = {"env_kwargs_override.goal_color": "Red"}


def shape(shape_dir):
    return {"eval_data": f"pushobj_eval/{shape_dir}/plan_targets.pkl"}


# "<task>__<name>" -> overrides; "eval_data" replaces the task's eval data
CONDITIONS = {
    # main tables (21)
    "mediummaze__default": {},
    "mediummaze__damping_50": {"env_kwargs_override.damping_scale": 50},
    "mediummaze__density_0.2": {"env_kwargs_override.density_scale": 0.2},
    "mediummaze__density_10": {"env_kwargs_override.density_scale": 10},
    "mediummaze__blur_2": BLUR,
    "diversemaze__unseen_3_5": {"min_block_radius": 3, "max_block_radius": 5},
    "pushshape__T": shape("val_T"),
    "pushshape__L": shape("val_L"),
    "pushshape__Z": shape("val_Z"),
    "pushshape__plus": shape("val_+"),
    "pushshape__I": shape("val_I"),
    "pushshape__small_tee": shape("val_small_tee"),
    "pushshape__square": shape("val_square"),
    "pusht__default": {},
    "pusht__blur_2": BLUR,
    "pusht__snp_0.01": {"ood_corruption": "snp1", "ood_level": 0.01},
    "pusht__dark_0.9": {"ood_corruption": "dark", "ood_level": 0.9},
    "pusht__red_agent": {"env_kwargs_override.agent_color": "Red"},
    "pusht__red_block": {"env_kwargs_override.color": "Red"},
    "pusht__red_anchor": RED_ANCHOR,
    "pusht__kv_2": KV,
    # compound shifts (7)
    "mediummaze__blur_2_damping_50": {**BLUR, "env_kwargs_override.damping_scale": 50},
    "mediummaze__blur_2_density_10": {**BLUR, "env_kwargs_override.density_scale": 10},
    "pusht__blur_2_kv_2": {**BLUR, **KV},
    "pusht__red_anchor_kv_2": {**RED_ANCHOR, **KV},
    "pusht__blur_2_red_anchor_kv_2": {**BLUR, **RED_ANCHOR, **KV},
    "pushshape__T_blur_2_kv_2": {**shape("val_T"), **BLUR, **KV},
    "pushshape__square_blur_2_kv_2": {**shape("val_square"), **BLUR, **KV},
}

# planner.adapt overrides on top of the config defaults
METHODS = {
    "frozen": {"lr": 0.0, "encoder_lr": 0.0, "steps": 0, "replay_buffer": False,
               "finetune_encoder": False},
    "predlast": {"predictor_block": "last"},
    "predfirst": {"predictor_block": "first"},
    "sandwich": {"mode": "residual"},
    "sandwich_noaction": {"mode": "residual", "streams": []},
    "sandwich_nooutput": {"mode": "residual", "output": "none"},
    "sandwich_noinput": {"mode": "residual", "streams": [], "encoder": "none"},
}

SUITES = {
    "main": (list(CONDITIONS), ["frozen", "predlast", "predfirst", "sandwich"]),
    "ablation": (["mediummaze__blur_2_damping_50", "pusht__blur_2_kv_2"],
                 ["sandwich_noaction", "sandwich_nooutput", "sandwich_noinput"]),
}


def hydra_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(map(str, value)) + "]"
    return str(value)


def command(seed, condition, method, run_dir, n_evals):
    config, model, eval_data = TASKS[condition.split("__")[0]]
    overrides = dict(CONDITIONS[condition])
    eval_data = overrides.pop("eval_data", eval_data)
    overrides.update({f"planner.adapt.{k}": v for k, v in METHODS[method].items()})
    return [
        sys.executable, "plan.py", "--config-name", config,
        f"ckpt_base_path={REPO / 'checkpoints' / model}",
        f"model_name={model}",
        f"eval_data_path={REPO / 'data' / eval_data}",
        f"n_evals={n_evals}",
        f"seed={seed}",
        "+wandb_logging=false",
        f"hydra.run.dir={run_dir}",
        *(f"++{k}={hydra_value(v)}" for k, v in overrides.items()),
    ]


def environment(task):
    """Thread counts and MuJoCo rendering backends of the paper runs."""
    env = dict(os.environ, OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", DATASET_DIR=str(REPO / "data"))
    libs = [env.get("LD_LIBRARY_PATH", ""), str(Path.home() / ".mujoco" / "mujoco210" / "bin")]
    if task.startswith("push"):
        env["MUJOCO_GL"] = "egl"
        env.pop("MUJOCO_PY_FORCE_CPU", None)
        libs.append("/usr/lib/nvidia")
    else:  # point maze renders through mujoco-py on the CPU
        env.pop("MUJOCO_GL", None)
        env["MUJOCO_PY_FORCE_CPU"] = "1"
        libs.append("/usr/lib/x86_64-linux-gnu")
    env["LD_LIBRARY_PATH"] = ":".join(x for x in libs if x)
    return env


def finished(run_dir):
    log = run_dir / "logs.json"
    return log.exists() and "final_eval/success_rate" in log.read_text()


def run(job):
    i, n, run_dir, cmd = job
    if finished(run_dir):
        print(f"[{i}/{n}] skip {run_dir.name}", flush=True)
        return True
    shutil.rmtree(run_dir, ignore_errors=True)  # logs.json is appended to, so start clean
    run_dir.mkdir(parents=True)
    print(f"[{i}/{n}] start {run_dir.name}", flush=True)
    with (run_dir / "run.log").open("w") as log:
        rc = subprocess.run(cmd, cwd=REPO, env=environment(run_dir.name.split("__")[1]),
                            stdout=log, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and finished(run_dir)
    print(f"[{i}/{n}] {'done' if ok else f'FAILED (rc={rc})'} {run_dir.name}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=list(SUITES), default="main")
    p.add_argument("--condition", action="append", choices=list(CONDITIONS), help="repeatable")
    p.add_argument("--method", action="append", choices=list(METHODS), help="repeatable")
    p.add_argument("--seed", action="append", type=int, help="repeatable")
    p.add_argument("--n_evals", type=int, default=N_EVALS)
    p.add_argument("--output", type=Path, help="default: outputs/<suite>")
    p.add_argument("--workers", type=int, default=1, help="runs in parallel")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    conditions, methods = SUITES[args.suite]
    output = (args.output or REPO / "outputs" / args.suite).resolve()
    runs = [(s, c, m) for s in args.seed or SEEDS for c in args.condition or conditions
            for m in args.method or methods]
    jobs = []
    for i, (s, c, m) in enumerate(runs, 1):
        run_dir = output / f"seed{s}__{c}__{m}"
        jobs.append((i, len(runs), run_dir, command(s, c, m, run_dir, args.n_evals)))
    if args.dry_run:
        for *_, cmd in jobs:
            print(" ".join(map(str, cmd)))
        return
    with ThreadPoolExecutor(args.workers) as pool:
        failed = sum(not ok for ok in pool.map(run, jobs))
    print(f"{len(jobs) - failed}/{len(jobs)} runs finished", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
