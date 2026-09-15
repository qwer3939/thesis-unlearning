"""
사용법
1. 실행: python filtering/detect_anomaly.py <평점 파일 경로>
2. MovieLens 100K(.data), MovieLens 1M(.dat), CSV(.csv)를 지원합니다.
3. 3종 탐지 규칙을 적용하고 suspicious_users.txt를 출력합니다.
"""

from pathlib import Path
import sys

import pandas as pd


# =========================
# 설정값
# =========================

INTERACTION_RULE = True

# 상호작용 수 z-score 임계값
INTERACTION_Z_THRESHOLD = 3.0

# Rule 2 ON / OFF
USE_RATING_VARIANCE_RULE = True

# 이 값 이하의 평점 분산을 이상으로 판단
RATING_VARIANCE_THRESHOLD = 0.1

# 평점을 너무 적게 남긴 사용자는 분산 판단에서 제외
MIN_INTERACTIONS_FOR_VARIANCE = 5

# Rule 3 ON / OFF
USE_POPULARITY_RULE = True

# 전체 아이템 중 인기 상위 몇 %를 인기 아이템으로 볼 것인지
POPULAR_ITEM_RATIO = 0.1

# 사용자의 interaction 중 인기 아이템 비율이
# 이 값 이상이면 의심 사용자로 판단
POPULARITY_THRESHOLD = 0.8

# 너무 적은 interaction을 가진 사용자는 제외
MIN_INTERACTIONS_FOR_POPULARITY = 5


def load_movielens():
    """MovieLens 100K 데이터를 불러옵니다."""

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "datasets"
        / "ml-100k"
        / "u.data"
    )

    ratings = pd.read_csv(
        data_path,
        sep="\t",
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp"
        ]
    )

    return ratings



def load_ratings(file_path):
    """
    평점 데이터를 불러옵니다.

    지원 형식:
    - MovieLens 100K (.data): 탭(\\t) 구분
    - MovieLens 1M (.dat):   :: 구분
    - CSV (.csv):            쉼표 구분

    필요한 컬럼:
    user_id, item_id, rating
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        ratings = pd.read_csv(file_path)

        required_columns = {
            "user_id",
            "item_id",
            "rating"
        }

        if not required_columns.issubset(ratings.columns):
            raise ValueError(
                "CSV 파일에는 user_id, item_id, rating "
                "컬럼이 필요합니다."
            )

    elif suffix == ".dat":
        # MovieLens 1M ratings.dat 형식:
        # user_id::item_id::rating::timestamp
        ratings = pd.read_csv(
            file_path,
            sep="::",
            engine="python",
            names=[
                "user_id",
                "item_id",
                "rating",
                "timestamp"
            ]
        )

    else:
        # MovieLens 100K u.data 형식:
        # user_id<TAB>item_id<TAB>rating<TAB>timestamp
        ratings = pd.read_csv(
            file_path,
            sep="\t",
            names=[
                "user_id",
                "item_id",
                "rating",
                "timestamp"
            ]
        )

    return ratings

def detect_interaction_anomaly(ratings, z_threshold=3.0):
    """
    Rule 1:
    사용자별 상호작용 수가 전체 사용자 분포에 비해
    지나치게 많은 사용자를 탐지합니다.
    """

    interaction_counts = (
        ratings
        .groupby("user_id")
        .size()
    )

    mean_count = interaction_counts.mean()
    std_count = interaction_counts.std()

    z_scores = (
        interaction_counts - mean_count
    ) / std_count

    suspicious_users = z_scores[
        z_scores > z_threshold
    ].index.tolist()

    return (
        suspicious_users,
        interaction_counts,
        z_scores,
        mean_count,
        std_count
    )


def detect_rating_variance_anomaly(
    ratings,
    variance_threshold=0.1,
    min_interactions=5
):
    """
    Rule 2
    사용자가 준 평점의 분산이 지나치게 낮으면 탐지합니다.

    예:
    5, 5, 5, 5, 5 -> variance = 0
    """

    user_stats = (
        ratings
        .groupby("user_id")["rating"]
        .agg(["count", "var"])
    )

    suspicious = user_stats[
        (user_stats["count"] >= min_interactions)
        & (user_stats["var"] <= variance_threshold)
    ]

    return suspicious


def detect_popularity_anomaly(
    ratings,
    popular_item_ratio=0.1,
    popularity_threshold=0.8,
    min_interactions=5
):
    """
    Rule 3
    사용자의 interaction이 인기 아이템에
    지나치게 집중되어 있는지 탐지합니다.
    """

    item_counts = (
        ratings
        .groupby("item_id")
        .size()
        .sort_values(ascending=False)
    )

    num_popular_items = max(
        1,
        int(len(item_counts) * popular_item_ratio)
    )

    popular_items = set(
        item_counts.head(num_popular_items).index
    )

    results = []

    for user_id, user_data in ratings.groupby("user_id"):

        total_interactions = len(user_data)

        if total_interactions < min_interactions:
            continue

        popular_count = (
            user_data["item_id"]
            .isin(popular_items)
            .sum()
        )

        popularity_ratio = (
            popular_count / total_interactions
        )

        if popularity_ratio >= popularity_threshold:

            results.append({
                "user_id": user_id,
                "interaction_count": total_interactions,
                "popular_count": popular_count,
                "popularity_ratio": popularity_ratio
            })

    return results, num_popular_items


def main():

    if len(sys.argv) < 2:
        print(
            "사용법: python filtering/detect_anomaly.py "
            "<평점 파일 경로>"
        )
        return

    input_path = sys.argv[1]

    ratings = load_ratings(input_path)

    # 모든 Rule의 탐지 결과를 통합해서 저장
    detected_users = {}

    print(f"입력 파일: {input_path}")

    print("===== Anomaly Detection =====")

    print(f"\n전체 평점 수: {len(ratings)}")
    print(
        f"전체 사용자 수: "
        f"{ratings['user_id'].nunique()}"
    )

    if INTERACTION_RULE:

        (
            suspicious_users,
            interaction_counts,
            z_scores,
            mean_count,
            std_count
        ) = detect_interaction_anomaly(
            ratings,
            INTERACTION_Z_THRESHOLD
        )

        print("\n===== Rule 1: Interaction Count =====")

        print(
            f"사용자 평균 상호작용 수: "
            f"{mean_count:.2f}"
        )

        print(
            f"표준편차: "
            f"{std_count:.2f}"
        )

        print(
            f"Z-score threshold: "
            f"{INTERACTION_Z_THRESHOLD}"
        )

        print(
            f"Rule 1 의심 사용자 수: "
            f"{len(suspicious_users)}"
        )

        print("\n의심 사용자:")

        for user_id in suspicious_users:
            detected_users.setdefault(user_id, []).append(
                "Interaction Count"
            )
            print(
                f"User {user_id} | "
                f"interactions="
                f"{interaction_counts[user_id]} | "
                f"z-score="
                f"{z_scores[user_id]:.2f}"
            )

    if USE_RATING_VARIANCE_RULE:

        rule2_result = detect_rating_variance_anomaly(
            ratings,
            RATING_VARIANCE_THRESHOLD,
            MIN_INTERACTIONS_FOR_VARIANCE
        )

        rule2_users = rule2_result.index.tolist()

        print("\n===== Rule 2: Rating Variance =====")

        print(
            f"Variance threshold: "
            f"{RATING_VARIANCE_THRESHOLD}"
        )

        print(
            f"Rule 2 의심 사용자 수: "
            f"{len(rule2_users)}"
        )

        print("\n의심 사용자:")

        for user_id in rule2_users:
            detected_users.setdefault(user_id, []).append(
                "Rating Variance"
            )

            print(
                f"User {user_id} | "
                f"interactions="
                f"{int(rule2_result.loc[user_id, 'count'])} | "
                f"variance="
                f"{rule2_result.loc[user_id, 'var']:.4f}"
            )

    if USE_POPULARITY_RULE:

        rule3_result, num_popular_items = (
            detect_popularity_anomaly(
                ratings,
                POPULAR_ITEM_RATIO,
                POPULARITY_THRESHOLD,
                MIN_INTERACTIONS_FOR_POPULARITY
            )
        )

        print(
            "\n===== Rule 3: Popularity Concentration ====="
        )

        print(
            f"인기 아이템 기준: 상위 "
            f"{POPULAR_ITEM_RATIO * 100:.0f}%"
        )

        print(
            f"인기 아이템 수: "
            f"{num_popular_items}"
        )

        print(
            f"Popularity threshold: "
            f"{POPULARITY_THRESHOLD}"
        )

        print(
            f"Rule 3 의심 사용자 수: "
            f"{len(rule3_result)}"
        )

        print("\n의심 사용자:")

        for result in rule3_result:
            user_id = result["user_id"]
            detected_users.setdefault(user_id, []).append(
                "Popularity Concentration"
            )

            print(
                f"User {result['user_id']} | "
                f"interactions={result['interaction_count']} | "
                f"popular={result['popular_count']} | "
                f"ratio={result['popularity_ratio']:.2f}"
            )

    print("\n===== Summary =====")

    total_users = ratings["user_id"].nunique()

    print(f"전체 사용자 수: {total_users}")
    print(f"전체 의심 사용자 수: {len(detected_users)}")

    if INTERACTION_RULE:
        print(
            f"Rule 1 검출 수: "
            f"{len(suspicious_users)}"
        )

    if USE_RATING_VARIANCE_RULE:
        print(
            f"Rule 2 검출 수: "
            f"{len(rule2_users)}"
        )

    if USE_POPULARITY_RULE:
        print(
            f"Rule 3 검출 수: "
            f"{len(rule3_result)}"
        )

    output_path = (
        Path(__file__).resolve().parent
        / "suspicious_users.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for user_id in sorted(detected_users):

            rules = ", ".join(
                detected_users[user_id]
            )

            f.write(
                f"{user_id}\t{rules}\n"
            )

    print(
        f"\n의심 사용자 목록 저장 완료: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
