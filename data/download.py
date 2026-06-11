"""
Downloads a labeled driving-scene image subset from SUN397 via HuggingFace.
SUN397 is a public scene recognition dataset with categories including highway,
parking_lot, street, etc. We filter to driving-relevant label IDs.

Saves images to data/images/ and writes data/labels.json.
"""

import json
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
IMAGES_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.json"

SAMPLES_PER_SCENE = 10

# SUN397 label index → our simplified scene label
# Indices confirmed from dataset features
LABEL_ID_MAP = {
    63:  "highway",      # bridge (highway overpass context)
    121: "city street",  # crosswalk
    138: "residential",  # driveway
    162: "highway",      # forest road
    169: "urban",        # gas station
    184: "highway",      # highway
    266: "parking lot",  # parking garage indoor
    267: "parking lot",  # parking garage outdoor
    268: "parking lot",  # parking lot
    295: "highway",      # railroad track (linear road-like)
    299: "residential",  # residential neighborhood
    335: "city street",  # street
}

# Simplified scene names for prompts
SCENE_CLASSES = sorted(set(LABEL_ID_MAP.values()))


def main():
    IMAGES_DIR.mkdir(exist_ok=True)

    print("Loading SUN397 from HuggingFace (streaming)...")
    ds = load_dataset("tanganke/sun397", split="test", streaming=True)

    records = []
    counts: dict[str, int] = defaultdict(int)

    for item in tqdm(ds, desc="Filtering driving scenes"):
        label_id = item["label"]
        scene = LABEL_ID_MAP.get(label_id)
        if scene is None:
            continue
        if counts[scene] >= SAMPLES_PER_SCENE:
            continue

        idx = sum(counts.values())
        fname = f"{idx:04d}_{scene.replace(' ', '_')}.jpg"
        img_path = IMAGES_DIR / fname
        item["image"].save(img_path)

        records.append({
            "file": fname,
            "scene": scene,
            "sun397_label_id": label_id,
            "timeofday": "daytime",
            "weather": "clear",
        })
        counts[scene] += 1

        if all(v >= SAMPLES_PER_SCENE for v in counts.values()) and len(counts) == len(SCENE_CLASSES):
            break

    with open(LABELS_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} images → {IMAGES_DIR}")
    print(f"Labels written → {LABELS_FILE}")
    print("\nScene distribution:")
    for scene, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {count}")


if __name__ == "__main__":
    main()
