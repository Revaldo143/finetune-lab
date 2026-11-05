"""Combine v1 (249) + v2 new (150) into 399 examples, then train/val split.

Mirrors build_dataset.py exactly: random.seed(42), shuffle all, split at
int(len * 0.95) -> 379 train / 20 val for 399.

v2 file layout (verified against the builder's rule-count breakdown):
rows 0-99  = 100 general examples
rows 100-149 = 50 style-focused examples (mapped to the 12 learnings.md rules)

Writes dataset_v2_train.jsonl / dataset_v2_val.jsonl.
v1 files (dataset_train.jsonl / dataset_val.jsonl) are NOT touched.
"""
import json
import random

V2_STYLE_START = 100  # first 100 v2 rows are general, last 50 are style


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    v1 = load("dataset_train.jsonl") + load("dataset_val.jsonl")
    v2 = load("dataset_v2_new.jsonl")
    assert len(v1) == 249, f"v1 expected 249, got {len(v1)}"
    assert len(v2) == 150, f"v2 expected 150, got {len(v2)}"

    rows = []
    for r in v1:
        rows.append({**r, "_group": "v1-general"})
    for i, r in enumerate(v2):
        rows.append({**r, "_group": "v2-style" if i >= V2_STYLE_START else "v2-general"})

    random.seed(42)
    random.shuffle(rows)
    split = int(len(rows) * 0.95)  # 379 for 399
    train, val = rows[:split], rows[split:]

    def group_counts(rs):
        c = {"v1-general": 0, "v2-general": 0, "v2-style": 0}
        for r in rs:
            c[r["_group"]] += 1
        return c

    print(f"train: {len(train)}, val: {len(val)}, total: {len(train) + len(val)}")
    print("train groups:", group_counts(train))
    print("val groups:", group_counts(val))
    print("val instructions:")
    for r in val:
        print(f"  - [{r['_group']}] {r['instruction']}")

    # strip the internal tag before writing
    for r in train + val:
        del r["_group"]

    with open("dataset_v2_train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("dataset_v2_val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
