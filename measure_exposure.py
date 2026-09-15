"""
사용법
1. 실행: python filtering/measure_exposure.py <추천 점수 파일> <target_items.txt>
2. 예시: python filtering/measure_exposure.py filtering/recommendation_scores.csv filtering/target_items.txt
3. 사용자별 Top-K에서 각 타깃 아이템의 Hit Ratio@K를 계산합니다.
"""

from pathlib import Path
import sys

import pandas as pd


# ==========================================
# 설정값
# ==========================================

K = 10


def load_recommendation_scores(file_path):
    """
    추천 점수 파일을 읽습니다.

    예상 형식:
    user_id,item_id,score
    """

    scores = pd.read_csv(file_path)

    required_columns = {
        "user_id",
        "item_id",
        "score"
    }

    if not required_columns.issubset(scores.columns):
        raise ValueError(
            "추천 점수 파일에는 "
            "user_id, item_id, score 컬럼이 필요합니다."
        )

    return scores


def load_target_items(file_path):
    """
    target_items.txt를 읽습니다.

    예:
    50
    100
    300
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        target_items = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    return target_items


def calculate_hr_at_k(
    scores,
    target_items,
    k=10
):
    """
    각 사용자의 Top-K 추천을 구한 뒤,
    타깃 아이템별 HR@K를 계산합니다.
    """

    # 사용자별 score가 높은 순서로 정렬
    sorted_scores = scores.sort_values(
        ["user_id", "score"],
        ascending=[True, False]
    )

    # 사용자별 Top-K
    top_k = (
        sorted_scores
        .groupby("user_id")
        .head(k)
    )

    total_users = scores["user_id"].nunique()

    results = {}

    for target_item in target_items:

        # Top-K 안에 target item이 등장한 사용자
        hit_users = (
            top_k[
                top_k["item_id"] == target_item
            ]["user_id"]
            .nunique()
        )

        hr = (
            hit_users / total_users
            if total_users > 0
            else 0.0
        )

        results[target_item] = {
            "hit_users": hit_users,
            "total_users": total_users,
            "hr": hr
        }

    return results


def main():

    if len(sys.argv) < 3:
        print(
            "사용법: python filtering/measure_exposure.py "
            "<추천 점수 파일> <target_items.txt>"
        )
        return

    scores_path = sys.argv[1]
    target_path = sys.argv[2]

    print(f"추천 점수 파일: {scores_path}")
    print(f"타깃 아이템 파일: {target_path}")

    scores = load_recommendation_scores(
        scores_path
    )

    target_items = load_target_items(
        target_path
    )

    results = calculate_hr_at_k(
        scores,
        target_items,
        K
    )

    print("\n===== Target Item Exposure =====")

    print(f"\nK: {K}")

    print(
        f"평가 사용자 수: "
        f"{scores['user_id'].nunique()}"
    )

    print(
        f"타깃 아이템 수: "
        f"{len(target_items)}"
    )

    print("\n===== HR@K Results =====")

    for target_item, result in results.items():

        print(
            f"Target Item {target_item} | "
            f"Hit Users: {result['hit_users']} / "
            f"{result['total_users']} | "
            f"HR@{K}: {result['hr']:.4f} "
            f"({result['hr'] * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
