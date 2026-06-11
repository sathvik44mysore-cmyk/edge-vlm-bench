"""
Benchmarks small VLMs on driving-scene classification tasks.

For each image in data/labels.json, sends the image + a prompt to each model
via Ollama's API, records latency and correctness, saves results to results/.

Usage:
    python benchmark.py [--models moondream llava-phi3] [--limit 20]
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import psutil
import requests
from tqdm import tqdm

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.json"
RESULTS_DIR = Path("results")

OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_MODELS = ["llava-phi3", "bakllava"]

TASKS = {
    "scene": {
        "prompt": (
            "Look at this image. "
            "Which single category best describes the road or driving environment shown? "
            "Choose exactly one of: highway, city street, residential. "
            'Respond with JSON in the form {"category": "<your choice>"}.'
        ),
        "classes": ["highway", "city street", "residential"],
        "gt_field": "scene",
    },
}


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_model(model: str, prompt: str, image_b64: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "format": "json",
        "stream": False,
    }
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    elapsed = time.perf_counter() - t0

    resp.raise_for_status()
    answer = resp.json().get("response", "").strip().lower()
    return answer, round(elapsed * 1000, 1)  # ms


def normalize_answer(raw: str, classes: list[str]) -> str:
    """Fuzzy-match the model's raw response to one of the valid class labels."""
    text = raw.lower().strip().rstrip(".")
    # JSON-mode responses look like {"category": "..."} — extract the value
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("category", text)).lower().strip()
    except (json.JSONDecodeError, ValueError):
        pass
    # Exact or substring match — prefer longer class names first to avoid partial hits
    for cls in sorted(classes, key=len, reverse=True):
        if cls in text:
            return cls
    # Token-level fallback: return the class with most word overlap
    tokens = set(text.split())
    best, best_score = "", 0
    for cls in classes:
        score = len(tokens & set(cls.split()))
        if score > best_score:
            best, best_score = cls, score
    return best if best_score > 0 else text[:40]


def stratified_sample(records: list[dict], limit: int) -> list[dict]:
    """Take records evenly across scene classes (round-robin) up to limit."""
    by_class: dict[str, list[dict]] = {}
    for rec in records:
        by_class.setdefault(rec["scene"], []).append(rec)
    sampled = []
    pools = list(by_class.values())
    i = 0
    while len(sampled) < limit and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            sampled.append(pool.pop(0))
        i += 1
    return sampled


def run_benchmark(models: list[str], limit: int | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(LABELS_FILE) as f:
        records = json.load(f)

    # Keep only images whose ground truth is one of the benchmark classes
    # ("urban" was dropped: it overlaps too heavily with "city street")
    valid_classes = set(TASKS["scene"]["classes"])
    records = [r for r in records if r["scene"] in valid_classes]

    if limit:
        records = stratified_sample(records, limit)

    print(f"Benchmarking {len(models)} model(s) on {len(records)} images\n")

    for model in models:
        print(f"=== {model} ===")
        results = []

        for rec in tqdm(records, desc=model):
            img_path = IMAGES_DIR / rec["file"]
            if not img_path.exists():
                continue

            image_b64 = encode_image(img_path)
            row = {"file": rec["file"], "scene_gt": rec["scene"], "tasks": {}}

            for task_name, task in TASKS.items():
                gt = rec.get(task["gt_field"], "")
                try:
                    answer, latency_ms = query_model(model, task["prompt"], image_b64)
                    predicted = normalize_answer(answer, task["classes"])
                    correct = predicted == gt.lower()
                    row["tasks"][task_name] = {
                        "predicted": predicted,
                        "gt": gt,
                        "correct": correct,
                        "latency_ms": latency_ms,
                        "raw": answer,
                    }
                except Exception as e:
                    row["tasks"][task_name] = {"error": str(e)}

            results.append(row)

        out_file = RESULTS_DIR / f"{model.replace(':', '_').replace('/', '_')}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        _print_summary(model, results)
        print()


def _print_summary(model: str, results: list[dict]):
    for task_name in TASKS:
        task_results = [
            r["tasks"][task_name]
            for r in results
            if task_name in r["tasks"] and "error" not in r["tasks"][task_name]
        ]
        if not task_results:
            continue
        correct = sum(1 for t in task_results if t["correct"])
        latencies = [t["latency_ms"] for t in task_results]
        avg_lat = sum(latencies) / len(latencies)
        print(
            f"  [{task_name}] accuracy: {correct}/{len(task_results)} "
            f"({100*correct/len(task_results):.1f}%)  "
            f"avg latency: {avg_lat:.0f}ms"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    args = parser.parse_args()
    run_benchmark(args.models, args.limit)


if __name__ == "__main__":
    main()
