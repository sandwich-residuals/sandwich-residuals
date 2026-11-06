"""Run the 48 OGBench-Cube experiments; finished runs are skipped."""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [1001, 1002, 1003]
CONDITIONS = ["default", "light_0.3", "camera_10", "gain_0.5"]
TAG = {"frozen": "frozen", "predlast": "predlast", "predfirst": "predfirst", "ours": "sandwich"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", action="append", choices=list(TAG))
    ap.add_argument("--condition", action="append", choices=CONDITIONS)
    ap.add_argument("--seed", action="append", type=int)
    ap.add_argument("--output", type=Path, default=HERE.parent / "outputs" / "cube")
    a = ap.parse_args()
    jobs = [(s, c, m) for s in a.seed or SEEDS for c in a.condition or CONDITIONS for m in a.method or list(TAG)]
    for i, (s, c, m) in enumerate(jobs, 1):
        out = a.output / f"seed{s}__cubedino__{c}__{TAG[m]}"
        if (out / "logs.json").exists():
            print(f"[{i}/{len(jobs)}] skip {out.name}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] start {out.name}", flush=True)
        rc = subprocess.run([sys.executable, HERE / "plan.py", "--method", m, "--condition", c, "--seed", str(s),
                             "--out", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        tail = (out / "run.log").read_text().splitlines()[-1] if (out / "run.log").exists() else ""
        print(f"[{i}/{len(jobs)}] end {out.name} rc={rc} {tail.split('cube ')[-1]}", flush=True)


if __name__ == "__main__":
    main()
