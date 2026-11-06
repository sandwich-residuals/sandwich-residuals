# Sandwich-Residuals: Parameter-Efficient Test-time Adaptation of World Models

Sandwich-Residuals adapts a pretrained latent world model to test-time distribution shifts
while keeping every pretrained weight frozen. Small residual modules around the predictor
(an action residual, an input visual residual and output residuals; about 0.1M parameters)
are learned online from the model's own prediction error.

This repository builds on the [AdaJEPA codebase](https://github.com/agentic-learning-ai-lab/adajepa).
The method is in [`planning/residual_adapt.py`](planning/residual_adapt.py) and is enabled
with `planner.adapt.mode=residual`.

## Setup

```bash
conda env create -f environment.yaml
conda activate ts
```

Download the AdaJEPA [checkpoints and eval data](https://drive.google.com/drive/folders/11IVDIrVDU6W47txR_ku1RkhRUj9F4ybs?usp=sharing)
and the [Medium Maze data](https://drive.google.com/drive/folders/1qbPO9MK7LwX2GBQq-fP_xXoy82wYlkde?usp=sharing)
into `checkpoints/` and `data/`.

## Running

Sandwich-Residuals on Medium Maze with 50× damping:

```bash
python plan.py --config-name adajepa_plan_cem_maze.yaml \
    ckpt_base_path=$(pwd)/checkpoints/mediummaze_dynamics_shift \
    model_name=mediummaze_dynamics_shift \
    eval_data_path=$(pwd)/data/point_maze_medium \
    seed=1001 +wandb_logging=false \
    ++planner.adapt.mode=residual \
    ++env_kwargs_override.damping_scale=50
```

To reproduce all experiments of the paper (seeds 1001–1005, 50 episodes each):

```bash
python run_experiments.py                    # main and compound-shift tables
python run_experiments.py --suite ablation   # residual-pathway ablation
python run_experiments.py --dry-run          # print the commands
```

The OGBench-Cube experiments with DINO-WM are in [`cube/`](cube/) and use a separate
environment:

```bash
pip install "stable-worldmodel[train,env,format]" scikit-learn
hf download quentinll/lewm-cube --repo-type dataset --local-dir data/lewm-cube
# extract data/lewm-cube/cube_single_expert.tar.zst to data/cube_single_expert.h5
python cube/run.py
```

`cube/run.py` expects the DINO-WM checkpoint in `checkpoints/cube_dinowm/`; to train it, see
[`cube/train/`](cube/train/).

## Results

Average success rate (%) over the conditions of each environment:

| Environment | Frozen | AdaJEPA PredLast | AdaJEPA PredFirst | Sandwich-Residuals |
|---|---:|---:|---:|---:|
| Medium Maze | 67.7 | 69.4 | 71.6 | **71.8** |
| Diverse Maze | 45.6 | 53.6 | 62.8 | **63.2** |
| PushObj | 38.4 | 64.2 | **66.2** | 62.3 |
| PushT | 47.5 | 60.0 | **68.6** | 63.4 |
| Compound shifts | 34.5 | 57.8 | 65.2 | **66.0** |
| Adapted parameters | 0 | 3.7M–9.9M | 3.7M–9.9M | 0.1M |

The logs of every run are in [`logs/`](logs/); `python results.py` prints all tables of the
paper from them.

## Acknowledgement

This code builds on [AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa),
[DINO-WM](https://github.com/gaoyuezhou/dino_wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).
