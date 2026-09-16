#!/usr/bin/env python3
"""
Generate real LightGCN Top-10 recommendation-score CSV files for HR@10 evaluation.

This replaces the test-only random score generator (make_example_scores.py).
It loads the saved Original / Retrain / IFRU checkpoints for attack ratios
0.01 / 0.03 / 0.05, scores candidate items for normal evaluation users, and
writes CSV files compatible with measure_exposure.py.

Important target-ID handling:
- attack_output/target_items.txt stores ORIGINAL MovieLens item IDs.
- processed CSVs use remapped internal item IDs.
- the current preprocessing did not save item_map.
- therefore this script identifies the internal target item from train_random.csv:
  the injected target is the item appearing for every fake user.
- if the mapping is ambiguous, the script stops instead of guessing.
"""

from pathlib import Path
import csv
import glob

import numpy as np
import pandas as pd
import torch

from utility.load_data import Data_for_LightGCN
from Model.Lightgcn import LightGCN


ATTACK_RATIOS = ("0.01", "0.03", "0.05")
MODEL_TYPES = ("original", "retrain", "ifru")
K = 10
PREDICT_BATCH_SIZE = 200_000

DATA_ROOT = Path("Data/Process/MovieLens")
TARGET_ORIGINAL_PATH = Path("attack_output/target_items.txt")
OUTPUT_DIR = Path("recommendation_scores")


class Args:
    def __init__(self, attack, data_type):
        self.lr = 1e-4
        self.regs = 0
        self.embed_size = 32
        self.batch_size = 2048
        self.epoch = 200
        self.data_path = "Data/Process/"
        self.dataset = "MovieLens"
        self.attack = attack
        self.layer_size = "[64,64]"
        self.verbose = 1
        self.Ks = "[10]"
        self.data_type = data_type
        self.init_std = 1e-4
        self.seed = 1024
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0


def read_original_targets():
    if not TARGET_ORIGINAL_PATH.is_file():
        raise FileNotFoundError(f"Target file not found: {TARGET_ORIGINAL_PATH}")
    return [
        int(x.strip())
        for x in TARGET_ORIGINAL_PATH.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def infer_internal_targets(ratio, expected_target_count):
    """Infer internal target IDs from fake-user coverage in train_random.csv."""
    path = DATA_ROOT / ratio / "train_random.csv"
    attack = pd.read_csv(path)
    required = {"user", "item", "label"}
    if not required.issubset(attack.columns):
        raise ValueError(f"Unexpected columns in {path}: {list(attack.columns)}")

    fake_user_count = attack["user"].nunique()
    if fake_user_count == 0:
        raise ValueError(f"No fake users found in {path}")

    coverage = (
        attack[["user", "item"]]
        .drop_duplicates()
        .groupby("item")["user"]
        .nunique()
    )
    candidates = sorted(
        int(item)
        for item, count in coverage.items()
        if int(count) == int(fake_user_count)
    )

    # If needed, restrict to positive interactions. The injected target is rated 5.
    if len(candidates) != expected_target_count:
        positive = attack[attack["label"] > 0][["user", "item"]].drop_duplicates()
        pos_coverage = positive.groupby("item")["user"].nunique()
        candidates = sorted(
            int(item)
            for item, count in pos_coverage.items()
            if int(count) == int(fake_user_count)
        )

    if len(candidates) != expected_target_count:
        top = coverage.sort_values(ascending=False).head(10)
        raise RuntimeError(
            "\nCould not identify the internal target item unambiguously.\n"
            f"ratio={ratio}, fake users={fake_user_count}, "
            f"expected targets={expected_target_count}, candidates={candidates}\n"
            f"Highest item coverages:\n{top}\n\n"
            "Do NOT continue with HR@10 until the target ID mapping is checked."
        )

    return candidates, fake_user_count


def find_checkpoint(model_type, ratio):
    if model_type == "original":
        patterns = [
            f"Weights/LightGCN/*data_type-full*attack-{ratio}*m.pth",
            f"Weights/LightGCN/*data_type-full*attack-{ratio}*.pth",
        ]
    elif model_type == "retrain":
        patterns = [f"Weights/LightGCN/*data_type-retraining*attack-{ratio}*.pth"]
    elif model_type == "ifru":
        patterns = [f"Weights/LightGCN_IFRU/*attack_{ratio}_*.pth"]
    else:
        raise ValueError(model_type)

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 checkpoint for {model_type} {ratio}, "
            f"found {len(matches)}:\n" + "\n".join(matches)
        )
    return Path(matches[0])


def build_data_and_model(model_type, ratio):
    # IFRU uses the contaminated full graph, matching the IFRU experiment.
    data_type = "retraining" if model_type == "retrain" else "full"
    args = Args(attack=ratio, data_type=data_type)

    data = Data_for_LightGCN(args, path=str(DATA_ROOT / ratio))
    data.set_train_mode(data_type)

    model = LightGCN(args, dataset=data).cuda()
    checkpoint = find_checkpoint(model_type, ratio)
    state = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict(state)
    model.eval()
    return data, model, checkpoint


def get_seen_positive_items(data):
    positive = data.train[data.train["label"] > 0][["user", "item"]]
    grouped = positive.groupby("user")["item"].apply(set)
    return {int(user): set(map(int, items)) for user, items in grouped.items()}


def predict_pairs(model, users, items):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(users), PREDICT_BATCH_SIZE):
            end = min(start + PREDICT_BATCH_SIZE, len(users))
            pred = model.predict(users[start:end], items[start:end])
            if torch.is_tensor(pred):
                pred = pred.detach().cpu().numpy()
            else:
                pred = np.asarray(pred)
            outputs.append(pred.reshape(-1))
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def generate_topk_csv(model_type, ratio, data, model, output_path):
    """Score candidates for normal test users and save only their Top-10 rows."""
    eval_users = np.sort(data.test["user"].unique().astype(np.int64))
    all_items = np.arange(int(data.n_items), dtype=np.int64)
    seen = get_seen_positive_items(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "score"])

        for idx, user in enumerate(eval_users, 1):
            blocked = seen.get(int(user), set())
            if blocked:
                mask = np.ones(len(all_items), dtype=bool)
                blocked_array = np.fromiter(
                    (x for x in blocked if 0 <= x < len(all_items)), dtype=np.int64
                )
                if blocked_array.size:
                    mask[blocked_array] = False
                candidate_items = all_items[mask]
            else:
                candidate_items = all_items

            if candidate_items.size == 0:
                continue

            users = np.full(candidate_items.size, int(user), dtype=np.int64)
            scores = predict_pairs(model, users, candidate_items)
            k = min(K, candidate_items.size)

            if k == candidate_items.size:
                top_idx = np.argsort(scores)[::-1]
            else:
                top_idx = np.argpartition(scores, -k)[-k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            for j in top_idx:
                writer.writerow([int(user), int(candidate_items[j]), float(scores[j])])
                total_rows += 1

            if idx % 500 == 0 or idx == len(eval_users):
                print(
                    f"  [{model_type} {ratio}] "
                    f"{idx}/{len(eval_users)} evaluation users processed"
                )

    return len(eval_users), total_rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this repository's LightGCN code.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_targets = read_original_targets()
    print("Original MovieLens target item IDs:", original_targets)

    mapping_rows = []

    print("\n===== 1. Target item ID mapping =====")
    for ratio in ATTACK_RATIOS:
        internal_targets, fake_users = infer_internal_targets(
            ratio, expected_target_count=len(original_targets)
        )

        internal_path = OUTPUT_DIR / f"target_items_internal_{ratio}.txt"
        internal_path.write_text(
            "".join(f"{item}\n" for item in internal_targets), encoding="utf-8"
        )

        for original_id, internal_id in zip(original_targets, internal_targets):
            mapping_rows.append(
                {
                    "attack_ratio": ratio,
                    "original_movielens_item_id": original_id,
                    "internal_item_id": internal_id,
                    "fake_users_after_preprocessing": fake_users,
                }
            )

        print(
            f"ratio {ratio}: original {original_targets} -> "
            f"internal {internal_targets} (fake users={fake_users})"
        )

    pd.DataFrame(mapping_rows).to_csv(
        OUTPUT_DIR / "target_item_mapping.csv", index=False
    )

    print("\n===== 2. Generate real model Top-10 scores =====")
    summary = []

    for ratio in ATTACK_RATIOS:
        for model_type in MODEL_TYPES:
            print(f"\n--- {model_type.upper()} / attack {ratio} ---")
            data, model, checkpoint = build_data_and_model(model_type, ratio)
            print("checkpoint:", checkpoint)

            output_path = OUTPUT_DIR / f"{model_type}_{ratio}_top10.csv"
            user_count, row_count = generate_topk_csv(
                model_type, ratio, data, model, output_path
            )

            target_file = OUTPUT_DIR / f"target_items_internal_{ratio}.txt"
            summary.append(
                {
                    "attack_ratio": ratio,
                    "model_type": model_type,
                    "checkpoint": str(checkpoint),
                    "evaluation_users": user_count,
                    "saved_rows": row_count,
                    "score_csv": str(output_path),
                    "target_file": str(target_file),
                }
            )

            print("saved:", output_path)
            print(
                "measure command:\n"
                f"  python measure_exposure.py {output_path} {target_file}"
            )

            del model
            del data
            torch.cuda.empty_cache()

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "generation_summary.csv", index=False
    )

    print("\n===== DONE =====")
    print(f"Outputs: {OUTPUT_DIR}/")
    print("Next: run measure_exposure.py on each generated score CSV.")


if __name__ == "__main__":
    main()
