"""Combine v2 combined (399) + v3 new (15) into 414 examples, then train/val split.

Mirrors build_dataset_v2.py exactly: random.seed(42), shuffle all, split at
int(len * 0.95) -> 393 train / 21 val for 414.

v3 file layout (verified):
rows 0-11   = 12 unknown-coin refusal examples (no 0x address in outputs)
rows 12-14  = 3 verified-coin positive examples (real STRK/COTI addresses)

Writes dataset_v3_train.jsonl / dataset_v3_val.jsonl.
v1 files, v2 files, and dataset_v3_new.jsonl are NOT touched.
"""
import json
import random

V3_REFUSAL_COUNT = 12  # first 12 v3 rows are refusals, last 3 are positives


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    v2 = load("dataset_v2_train.jsonl") + load("dataset_v2_val.jsonl")
    v3 = load("dataset_v3_new.jsonl")
    assert len(v2) == 399, f"v2 expected 399, got {len(v2)}"
    assert len(v3) == 15, f"v3 expected 15, got {len(v3)}"

    rows = []
    for r in v2:
        rows.append({**r, "_group": "v2"})
    for i, r in enumerate(v3):
        rows.append({**r, "_group": "v3-refusal" if i < V3_REFUSAL_COUNT else "v3-positive"})

    random.seed(42)
    random.shuffle(rows)
    split = int(len(rows) * 0.95)  # 393 for 414
    train, val = rows[:split], rows[split:]

    def group_counts(rs):
        c = {"v2": 0, "v3-refusal": 0, "v3-positive": 0}
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

    with open("dataset_v3_train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("dataset_v3_val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
