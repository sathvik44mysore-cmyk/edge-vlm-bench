"""
Reads benchmark results from results/ and generates charts in charts/.

Charts produced:
  1. accuracy_vs_latency.png  — scatter: avg latency vs accuracy per model
  2. per_scene_accuracy.png   — grouped bar: accuracy by scene class per model
  3. latency_distribution.png — box plot: latency spread per model
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
CHARTS_DIR = Path("charts")

MODEL_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]


def load_results() -> dict[str, list[dict]]:
    data = {}
    for path in RESULTS_DIR.glob("*.json"):
        model_name = path.stem.replace("_", ":", 1) if ":" not in path.stem else path.stem
        with open(path) as f:
            data[model_name] = json.load(f)
    return data


def compute_stats(results: list[dict]) -> dict:
    task = "scene"
    rows = [
        r["tasks"][task]
        for r in results
        if task in r.get("tasks", {}) and "error" not in r["tasks"][task]
    ]
    if not rows:
        return {}

    correct = [r["correct"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]

    by_scene: dict[str, list] = defaultdict(list)
    for r_full in results:
        t = r_full.get("tasks", {}).get(task, {})
        if "error" in t:
            continue
        by_scene[r_full["scene_gt"]].append(t["correct"])

    scene_acc = {s: sum(v) / len(v) for s, v in by_scene.items() if v}

    return {
        "accuracy": sum(correct) / len(correct),
        "avg_latency_ms": np.mean(latencies),
        "latencies": latencies,
        "scene_accuracy": scene_acc,
        "n": len(rows),
    }


def plot_accuracy_vs_latency(all_stats: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (model, stats) in enumerate(all_stats.items()):
        if not stats:
            continue
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.scatter(
            stats["avg_latency_ms"] / 1000,
            stats["accuracy"] * 100,
            s=200,
            color=color,
            zorder=5,
            label=model,
        )
        ax.annotate(
            model,
            (stats["avg_latency_ms"] / 1000, stats["accuracy"] * 100),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Average Latency (seconds)", fontsize=11)
    ax.set_ylabel("Scene Classification Accuracy (%)", fontsize=11)
    ax.set_title("Edge VLM Benchmark: Accuracy vs Latency\n(driving scene classification)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = CHARTS_DIR / "accuracy_vs_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_per_scene_accuracy(all_stats: dict[str, dict]):
    all_scenes = sorted(
        set(s for stats in all_stats.values() for s in stats.get("scene_accuracy", {}))
    )
    if not all_scenes:
        return

    models = list(all_stats.keys())
    x = np.arange(len(all_scenes))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        accs = [
            all_stats[model].get("scene_accuracy", {}).get(scene, 0) * 100
            for scene in all_scenes
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        bars = ax.bar(x + offset, accs, width * 0.9, label=model, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_scenes, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Per-Scene Accuracy by Model", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = CHARTS_DIR / "per_scene_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_latency_distribution(all_stats: dict[str, dict]):
    models = [m for m, s in all_stats.items() if s.get("latencies")]
    latencies = [
        [v / 1000 for v in all_stats[m]["latencies"]] for m in models
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(latencies, tick_labels=models, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], MODEL_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title("Inference Latency Distribution per Model", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    out = CHARTS_DIR / "latency_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    CHARTS_DIR.mkdir(exist_ok=True)
    all_results = load_results()

    if not all_results:
        print("No results found in results/. Run benchmark.py first.")
        return

    all_stats = {model: compute_stats(results) for model, results in all_results.items()}

    print("\nSummary:")
    for model, stats in all_stats.items():
        if stats:
            print(
                f"  {model}: accuracy={stats['accuracy']*100:.1f}%  "
                f"avg_latency={stats['avg_latency_ms']:.0f}ms  n={stats['n']}"
            )

    plot_accuracy_vs_latency(all_stats)
    plot_per_scene_accuracy(all_stats)
    plot_latency_distribution(all_stats)
    print("\nAll charts saved to charts/")


if __name__ == "__main__":
    main()
