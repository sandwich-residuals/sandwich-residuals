"""Print the paper's tables, computed from the final success rates in logs/."""

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN_SEEDS = [1001, 1002, 1003, 1004, 1005]
CUBE_SEEDS = [1001, 1002, 1003]
ARMS = ["frozen", "predlast", "predfirst", "sandwich"]
ABLATIONS = ["sandwich_noaction", "sandwich_nooutput", "sandwich_noinput"]
HEADER = "| Frozen | PredLast | PredFirst | Ours |"

MAIN = {
    "Medium Maze": [
        ("mediummaze__default", "No shift"),
        ("mediummaze__damping_50", "Damping 50×"),
        ("mediummaze__density_0.2", "Density 0.2×"),
        ("mediummaze__density_10", "Density 10×"),
        ("mediummaze__blur_2", "Blur 2×"),
    ],
    "Diverse Maze": [("diversemaze__unseen_3_5", "Unseen layouts")],
    "PushObj": [
        ("pushshape__T", "T"),
        ("pushshape__L", "L"),
        ("pushshape__Z", "Z"),
        ("pushshape__plus", "Plus"),
        ("pushshape__I", "I (unseen)"),
        ("pushshape__small_tee", "Small T (unseen)"),
        ("pushshape__square", "Square (unseen)"),
    ],
    "PushT": [
        ("pusht__default", "No shift"),
        ("pusht__blur_2", "Blur 2×"),
        ("pusht__snp_0.01", "Salt-and-pepper 0.01"),
        ("pusht__dark_0.9", "Brightness 0.9×"),
        ("pusht__red_agent", "Red agent"),
        ("pusht__red_block", "Red block"),
        ("pusht__red_anchor", "Red anchor"),
        ("pusht__kv_2", "Controller `kv` 2×"),
    ],
}
COMPOUND = [
    ("mediummaze__blur_2_damping_50", "Medium Maze: blur 2× + damping 50×"),
    ("mediummaze__blur_2_density_10", "Medium Maze: blur 2× + density 10×"),
    ("pusht__blur_2_kv_2", "PushT: blur 2× + `kv` 2×"),
    ("pusht__red_anchor_kv_2", "PushT: red anchor + `kv` 2×"),
    ("pusht__blur_2_red_anchor_kv_2", "PushT: blur 2× + red anchor + `kv` 2×"),
    ("pushshape__T_blur_2_kv_2", "PushObj T: blur 2× + `kv` 2×"),
    ("pushshape__square_blur_2_kv_2", "PushObj Square: blur 2× + `kv` 2×"),
]
CUBE = [
    ("cubedino__default", "No shift"),
    ("cubedino__light_0.3", "Light 0.3"),
    ("cubedino__camera_10", "Camera 10°"),
    ("cubedino__gain_0.5", "Arm gain 0.5×"),
]
ABLATION = [
    ("sandwich", "Full"),
    ("sandwich_noaction", "Without action residual"),
    ("sandwich_nooutput", "Without output residuals"),
    ("sandwich_noinput", "Without input residuals"),
]
DYNAMICS = {"mediummaze__damping_50", "mediummaze__density_0.2", "mediummaze__density_10", "pusht__kv_2"}


def load_runs():
    """(condition, arm) -> {seed: success %}."""
    runs = defaultdict(dict)
    arms = sorted(ARMS + ABLATIONS, key=len, reverse=True)  # longest first: "sandwich_noaction" before "sandwich"
    pat = re.compile(r"seed(\d+)__(.+)__(" + "|".join(arms) + r")$")
    for log in sorted(ROOT.glob("logs/*/seed*/logs.json")):
        m = pat.match(log.parent.name)
        recs = [json.loads(line) for line in log.read_text().splitlines() if "final_eval/success_rate" in line]
        if m and recs:
            runs[(m[2], m[3])][int(m[1])] = 100 * recs[-1]["final_eval/success_rate"]
    return runs


def bold_best(cells, means):
    best = max(means)
    return [f"**{c}**" if abs(m - best) < 1e-9 else c for c, m in zip(cells, means)]


def row(runs, cond, seeds):
    """Cells 'mean ± sd' for ARMS (best in bold) and the means."""
    vals = [[runs[(cond, a)][s] for s in seeds] for a in ARMS]
    means = [st.mean(v) for v in vals]
    return bold_best([f"{st.mean(v):.1f} ± {st.stdev(v):.1f}" for v in vals], means), means


def avg_row(means_list):
    means = [st.mean(m[i] for m in means_list) for i in range(len(ARMS))]
    return bold_best([f"{m:.1f}" for m in means], means)


def paired(runs, base, pairs):
    d = [runs[(c, "sandwich")][s] - runs[(c, base)][s] for c, s in pairs]
    return f"{st.mean(d):+.1f} ± {st.stdev(d) / len(d) ** 0.5:.1f}"


def build():
    runs = load_runs()
    out = ["# Results", "",
           "Success rate (%), mean ± standard deviation over seeds of 50 episodes each "
           "(seeds 1001–1005; 1001–1003 for OGBench-Cube). Best per row in bold. "
           "Computed from `logs/`.", ""]

    out += ["## Table I: main results", "", "| Environment | Condition " + HEADER, "|---|---|" + "---:|" * 4]
    all_means = []
    for env, rows in MAIN.items():
        env_means = []
        for i, (cond, label) in enumerate(rows):
            cells, means = row(runs, cond, MAIN_SEEDS)
            env_means.append(means)
            out.append(f"| {env if i == 0 else ''} | {label} | " + " | ".join(cells) + " |")
        if len(rows) > 1:
            out.append("| | *Average* | " + " | ".join(avg_row(env_means)) + " |")
        all_means += env_means
    out += [f"| **All {len(all_means)} conditions** | | " + " | ".join(avg_row(all_means)) + " |", ""]

    out += ["## Table II: compound shifts", "", "| Condition " + HEADER, "|---|" + "---:|" * 4]
    comp_means = []
    for cond, label in COMPOUND:
        cells, means = row(runs, cond, MAIN_SEEDS)
        comp_means.append(means)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["| **Average** | " + " | ".join(avg_row(comp_means)) + " |", ""]

    main_pairs = [(c, s) for rows in MAIN.values() for c, _ in rows for s in MAIN_SEEDS]
    subsets = [("All 21 conditions", main_pairs)]
    subsets += [(env, [(c, s) for c, s in main_pairs if c in dict(rows)]) for env, rows in MAIN.items()]
    subsets += [("Compound (7)", [(c, s) for c, _ in COMPOUND for s in MAIN_SEEDS]),
                ("OGBench-Cube", [(c, s) for c, _ in CUBE for s in CUBE_SEEDS])]
    out += ["## Table III: paired differences", "",
            "Ours minus each baseline, paired by (condition, seed); mean ± standard error.", "",
            "| Subset | − Frozen | − PredLast | − PredFirst |", "|---|---:|---:|---:|"]
    for name, pairs in subsets:
        out.append(f"| {name} | " + " | ".join(paired(runs, b, pairs) for b in ARMS[:3]) + " |")
    dyn = [p for p in main_pairs if p[0] in DYNAMICS]
    other = [p for p in main_pairs if p[0] not in DYNAMICS]
    cube_shifted = [(c, s) for c, _ in CUBE[1:] for s in CUBE_SEEDS]
    out += ["", f"Ours − PredFirst: {paired(runs, 'predfirst', dyn)} on dynamics shifts, "
                f"{paired(runs, 'predfirst', other)} on the other conditions. "
                f"OGBench-Cube shifted conditions, Ours − Frozen: {paired(runs, 'frozen', cube_shifted)}.", ""]

    out += ["## Table IV: OGBench-Cube (DINO-WM)", "", "| Condition " + HEADER, "|---|" + "---:|" * 4]
    cube_means = []
    for cond, label in CUBE:
        cells, means = row(runs, cond, CUBE_SEEDS)
        cube_means.append(means)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["| **Average** | " + " | ".join(avg_row(cube_means)) + " |", ""]

    out += ["## Table V: residual-pathway ablation", "",
            "Compound shifts; parentheses give the change from the full method.", "",
            "| Configuration | Medium Maze: blur 2× + damping 50× | PushT: blur 2× + `kv` 2× |", "|---|---:|---:|"]
    conds = ["mediummaze__blur_2_damping_50", "pusht__blur_2_kv_2"]
    full = {c: st.mean(runs[(c, "sandwich")][s] for s in MAIN_SEEDS) for c in conds}
    for arm, label in ABLATION:
        cells = []
        for c in conds:
            v = [runs[(c, arm)][s] for s in MAIN_SEEDS]
            diff = "" if arm == "sandwich" else f" ({st.mean(v) - full[c]:+.1f})"
            cells.append(f"{st.mean(v):.1f} ± {st.stdev(v):.1f}{diff}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(build(), end="")
