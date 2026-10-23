"""Combine all parts into dataset.jsonl (Alpaca format) + train/val split."""
import json
import random

from data.part1_persona import EXAMPLES as p1
from data.part2_projects import EXAMPLES as p2
from data.part3_preferences import EXAMPLES as p3
from data.part4_general import EXAMPLES as p4

all_ex = p1 + p2 + p3 + p4
random.seed(42)
random.shuffle(all_ex)

rows = [{"instruction": i, "input": "", "output": o} for i, o in all_ex]
split = int(len(rows) * 0.95)

with open("dataset_train.jsonl", "w", encoding="utf-8") as f:
    for r in rows[:split]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("dataset_val.jsonl", "w", encoding="utf-8") as f:
    for r in rows[split:]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train: {split}, val: {len(rows) - split}, total: {len(rows)}")
