#!/usr/bin/env python3
"""
SISA / RecEraser Top-10 추천 점수 생성 스크립트.

generate_recommendation_scores_final.py와 같은 형식의 CSV를 만들되,
모델이 RecEraser_LightGCN 계열이라 데이터 로더와 모델 클래스가 다르다.

- SISA     : part_type=3 (random partition)   -> Weights/LightGCN_SISA/
- RecEraser: part_type=1 (interaction based)  -> Weights/LightGCN_Eraser/

두 모델 모두 retraining 모드(공격 데이터 제외)로 학습되었으므로
data_type='retraining'으로 그래프를 구성한다.

출력: recommendation_scores/{sisa,eraser}_{ratio}_top10.csv
이후 measure_exposure.py로 HR@10을 계산한다.
"""

from pathlib import Path
import csv
import glob
import os

import numpy as np
import pandas as pd
import torch

from utility.load_data import Data_for_RecEraser_LightGCN
from Model.Eraser import RecEraser_LightGCN


ATTACK_RATIOS = ("0.01", "0.03", "0.05")
MODEL_TYPES = ("sisa", "eraser")
K = 10
PREDICT_BATCH_SIZE = 200_000

N_FILLERS = int(os.environ.get("N_FILLERS", 150))
DATA_ROOT = Path("Data/Process/MovieLens")
OUTPUT_DIR = Path("recommendation_scores")

# part_type: SISA=3(random), RecEraser=1(interaction based)
PART_TYPE = {"sisa": 3, "eraser": 1}
WEIGHT_DIR = {"sisa": "Weights/LightGCN_SISA", "eraser": "Weights/LightGCN_Eraser"}


class Args:
    """RecEraser_LightGCN 생성에 필요한 하이퍼파라미터.

    학습 당시(sisa/eraser_lightgcn_movie_server_final.py)의 config와
    반드시 동일해야 state_dict 로드가 성공한다.
    """

    def __init__(self, attack, part_type):
        self.lr = 1e-4
        self.regs = 0
        self.regs_agg = 0
        self.embed_size = 32
        self.batch_size = 2048
        self.epoch = 200
        self.data_path = "Data/Process/"
        self.dataset = "MovieLens"
        self.attack = attack
        self.pretrain = 0
        self.n_layers = 1
        self.verbose = 1
        self.data_type = "retraining"
        self.save_flag = 1
        self.drop_prob = 0
        self.biased = False
        self.init_std = 1e-4
        self.part_type = part_type
        self.part_num = 10
        self.part_T = 50
        self.n_fillers = N_FILLERS
        self.seed = 1024


def find_checkpoint(model_type, ratio):
    """학습 스크립트가 만든 파일명 규칙에 맞춰 체크포인트를 찾는다."""
    pattern = (
        f"{WEIGHT_DIR[model_type]}/*attack_{ratio}_*"
        f"n_fillers_{N_FILLERS}_part_type_{PART_TYPE[model_type]}_-m.pth"
    )
    matches = sorted(set(glob.glob(pattern)))

    if len(matches) != 1:
        raise RuntimeError(
            f"{model_type} {ratio}: 체크포인트가 1개여야 하는데 "
            f"{len(matches)}개 발견됨.\n패턴: {pattern}\n"
            + "\n".join(matches)
        )
    return Path(matches[0])


def build_data_and_model(model_type, ratio):
    part_type = PART_TYPE[model_type]
    args = Args(attack=ratio, part_type=part_type)

    data = Data_for_RecEraser_LightGCN(
        str(DATA_ROOT / ratio),
        args.batch_size,
        args.part_type,
        args.part_num,
        args.part_T,
    )
    data.set_train_mode(args.data_type)

    config = {"n_users": data.n_users, "n_items": data.n_items}
    model = RecEraser_LightGCN(config, args).cuda()
    model.Graph = data.Graph

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
        raise RuntimeError("CUDA GPU가 필요합니다.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"필러 수: {N_FILLERS}")

    # 타깃 내부 ID는 generate_recommendation_scores_final.py가 이미 저장해 둔 것을 재사용한다.
    for ratio in ATTACK_RATIOS:
        target_file = OUTPUT_DIR / f"target_items_internal_{ratio}.txt"
        if not target_file.is_file():
            raise FileNotFoundError(
                f"{target_file} 없음. "
                "generate_recommendation_scores_final.py를 먼저 실행하세요."
            )

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
        OUTPUT_DIR / f"generation_summary_sisa_eraser_filler{N_FILLERS}.csv",
        index=False,
    )

    print("\n===== DONE =====")
    print(f"출력 위치: {OUTPUT_DIR}/")
    print("다음: measure_exposure.py로 HR@10 측정")


if __name__ == "__main__":
    main()
