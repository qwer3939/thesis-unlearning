#!/usr/bin/env python3
"""
언러닝 6종 모델의 valid / test AUC 일괄 측정.

Original / Retrain / IFRU / SCIF  -> LightGCN 클래스
SISA / RecEraser                  -> RecEraser_LightGCN 클래스

두 계열은 모델 구조와 데이터 로더가 다르므로 내부에서 분기한다.
저장된 .pth를 불러와 test/valid set에 대한 AUC만 계산하며,
학습이나 언러닝은 수행하지 않는다.

실행:
    CUDA_VISIBLE_DEVICES=2 python measure_auc_all.py
    N_FILLERS=200 CUDA_VISIBLE_DEVICES=2 python measure_auc_all.py

출력:
    results/auc_all_filler{N}.csv  +  터미널 비교표
"""

from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from utility.load_data import Data_for_LightGCN, Data_for_RecEraser_LightGCN
from Model.Lightgcn import LightGCN
from Model.Eraser import RecEraser_LightGCN


ATTACK_RATIOS = ("0.01", "0.03", "0.05")
MODEL_TYPES = ("original", "retrain", "ifru", "scif", "sisa", "eraser")

N_FILLERS = int(os.environ.get("N_FILLERS", 150))
DATA_ROOT = Path("Data/Process/MovieLens")
RESULT_DIR = Path("results")

# RecEraser 계열 설정
PART_TYPE = {"sisa": 3, "eraser": 1}
WEIGHT_DIR = {"sisa": "Weights/LightGCN_SISA", "eraser": "Weights/LightGCN_Eraser"}

# 학습 시 data_type. SISA/RecEraser는 학습 스크립트 config가 retraining이었음.
DATA_TYPE = {
    "original": "full",
    "retrain": "retraining",
    "ifru": "full",
    "scif": "full",
    "sisa": "retraining",
    "eraser": "retraining",
}


class ArgsLightGCN:
    """LightGCN 계열(original/retrain/ifru/scif)용 하이퍼파라미터."""

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
        self.n_fillers = N_FILLERS


class ArgsEraser:
    """RecEraser 계열(sisa/eraser)용 하이퍼파라미터.

    학습 당시 config와 동일해야 state_dict 로드가 성공한다.
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
    if model_type == "original":
        patterns = [
            f"Weights/LightGCN/*data_type-full*attack-{ratio}*n_fillers-{N_FILLERS}*m.pth"
        ]
    elif model_type == "retrain":
        patterns = [
            f"Weights/LightGCN/*data_type-retraining*attack-{ratio}*n_fillers-{N_FILLERS}*.pth"
        ]
    elif model_type == "ifru":
        patterns = [f"Weights/LightGCN_IFRU/*filler_{N_FILLERS}*attack_{ratio}_*.pth"]
    elif model_type == "scif":
        patterns = [f"Weights/LightGCN_SCIF/*filler_{N_FILLERS}*attack_{ratio}_*.pth"]
    elif model_type in ("sisa", "eraser"):
        patterns = [
            f"{WEIGHT_DIR[model_type]}/*attack_{ratio}_*"
            f"n_fillers_{N_FILLERS}_part_type_{PART_TYPE[model_type]}_-m.pth"
        ]
    else:
        raise ValueError(model_type)

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))

    if len(matches) != 1:
        raise RuntimeError(
            f"{model_type} {ratio}: 체크포인트가 1개여야 하는데 {len(matches)}개 발견됨.\n"
            + "\n".join(matches)
        )
    return Path(matches[0])


def build_model(model_type, ratio):
    """모델 계열에 따라 데이터 로더와 모델 클래스를 분기해서 로드한다."""
    if model_type in ("sisa", "eraser"):
        args = ArgsEraser(attack=ratio, part_type=PART_TYPE[model_type])
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
    else:
        data_type = DATA_TYPE[model_type]
        args = ArgsLightGCN(attack=ratio, data_type=data_type)
        data = Data_for_LightGCN(args, path=str(DATA_ROOT / ratio))
        data.set_train_mode(data_type)
        model = LightGCN(args, dataset=data).cuda()

    checkpoint = find_checkpoint(model_type, ratio)
    state = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict(state)
    model.eval()
    return data, model, checkpoint


def compute_auc(model, data):
    """test / valid AUC를 계산한다."""
    results = {}
    for split in ("valid", "test"):
        df = getattr(data, split)[["user", "item", "label"]].values
        with torch.no_grad():
            pred = model.predict(df[:, 0], df[:, 1])
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.asarray(pred)
        results[f"{split}_auc"] = roc_auc_score(df[:, -1], pred.reshape(-1))
    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"필러 수: {N_FILLERS}\n")

    rows = []

    for ratio in ATTACK_RATIOS:
        for model_type in MODEL_TYPES:
            print(f"--- {model_type.upper()} / attack {ratio} ---", flush=True)
            try:
                data, model, checkpoint = build_model(model_type, ratio)
                auc = compute_auc(model, data)

                print(f"  checkpoint : {checkpoint.name}")
                print(f"  valid AUC  : {auc['valid_auc']:.6f}")
                print(f"  test  AUC  : {auc['test_auc']:.6f}\n", flush=True)

                rows.append(
                    {
                        "attack_ratio": ratio,
                        "model_type": model_type,
                        "data_type": DATA_TYPE[model_type],
                        "valid_auc": round(auc["valid_auc"], 6),
                        "test_auc": round(auc["test_auc"], 6),
                        "checkpoint": checkpoint.name,
                    }
                )

                del model, data
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [실패] {e}\n", flush=True)
                rows.append(
                    {
                        "attack_ratio": ratio,
                        "model_type": model_type,
                        "data_type": DATA_TYPE[model_type],
                        "valid_auc": None,
                        "test_auc": None,
                        "checkpoint": f"ERROR: {e}",
                    }
                )

    df = pd.DataFrame(rows)
    out_path = RESULT_DIR / f"auc_all_filler{N_FILLERS}.csv"
    df.to_csv(out_path, index=False)

    # 터미널 비교표
    print("\n" + "=" * 72)
    print(f"Test AUC 비교표 (필러 {N_FILLERS})")
    print("=" * 72)
    pivot = df.pivot(index="attack_ratio", columns="model_type", values="test_auc")
    pivot = pivot.reindex(columns=[m for m in MODEL_TYPES if m in pivot.columns])
    print(pivot.to_string())

    print("\n" + "=" * 72)
    print(f"Valid AUC 비교표 (필러 {N_FILLERS})")
    print("=" * 72)
    pivot_v = df.pivot(index="attack_ratio", columns="model_type", values="valid_auc")
    pivot_v = pivot_v.reindex(columns=[m for m in MODEL_TYPES if m in pivot_v.columns])
    print(pivot_v.to_string())

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
