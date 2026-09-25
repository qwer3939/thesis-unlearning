from pathlib import Path
import pandas as pd

from detect_anomaly import load_ratings, detect_popularity_anomaly


# 실험할 오염률
ATTACK_RATIOS = ["0.01", "0.03", "0.05"]

# Rule 3 임계값: 70%, 80%, 90%
THRESHOLDS = [0.7, 0.8, 0.9]

# detect_anomaly.py와 동일한 설정
POPULAR_ITEM_RATIO = 0.1
MIN_INTERACTIONS = 5


def load_fake_users(file_path):
    """실제로 주입된 가짜 사용자 ID를 읽는다."""
    with open(file_path, "r", encoding="utf-8") as f:
        return {
            int(line.strip())
            for line in f
            if line.strip()
        }


def main():

    project_root = Path(__file__).resolve().parent
    attack_dir = project_root / "attack_output"

    results = []

    print("===== Rule 3 Threshold Experiment =====")

    for attack_ratio in ATTACK_RATIOS:

        ratings_path = (
            attack_dir / f"ratings_attacked_{attack_ratio}.dat"
        )

        fake_users_path = (
            attack_dir / f"fake_users_{attack_ratio}.txt"
        )

        if not ratings_path.exists():
            raise FileNotFoundError(
                f"공격 데이터가 없습니다: {ratings_path}"
            )

        if not fake_users_path.exists():
            raise FileNotFoundError(
                f"가짜 사용자 목록이 없습니다: {fake_users_path}"
            )

        ratings = load_ratings(ratings_path)
        fake_users = load_fake_users(fake_users_path)

        all_users = set(ratings["user_id"].unique())
        normal_users = all_users - fake_users

        print()
        print(f"===== Attack Ratio {attack_ratio} =====")
        print(f"전체 사용자 수: {len(all_users)}")
        print(f"가짜 사용자 수: {len(fake_users)}")
        print(f"정상 사용자 수: {len(normal_users)}")

        for threshold in THRESHOLDS:

            rule3_result, num_popular_items = (
                detect_popularity_anomaly(
                    ratings,
                    popular_item_ratio=POPULAR_ITEM_RATIO,
                    popularity_threshold=threshold,
                    min_interactions=MIN_INTERACTIONS
                )
            )

            detected_users = {
                result["user_id"]
                for result in rule3_result
            }

            # True Positive:
            # 실제 가짜 사용자 중 Rule 3가 탐지한 사용자
            true_positives = detected_users & fake_users

            # False Positive:
            # 정상 사용자인데 Rule 3가 잘못 탐지한 사용자
            false_positives = detected_users & normal_users

            # False Negative:
            # 실제 가짜 사용자인데 탐지하지 못한 사용자
            false_negatives = fake_users - detected_users

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            # 정상 사용자 중 잘못 탐지된 비율
            fpr = (
                fp / len(normal_users)
                if len(normal_users) > 0
                else 0
            )

            # 가짜 사용자 탐지율
            tpr = (
                tp / len(fake_users)
                if len(fake_users) > 0
                else 0
            )

            print()
            print(f"Threshold: {threshold:.1f}")
            print(f"Rule 3 검출 수: {len(detected_users)}")
            print(f"TP (가짜 사용자 검출): {tp}")
            print(f"FP (정상 사용자 오탐): {fp}")
            print(f"FN (가짜 사용자 미탐): {fn}")
            print(f"TPR (탐지율): {tpr:.4f} ({tpr * 100:.2f}%)")
            print(f"FPR (오탐률): {fpr:.4f} ({fpr * 100:.2f}%)")

            results.append({
                "attack_ratio": attack_ratio,
                "threshold": threshold,
                "total_users": len(all_users),
                "fake_users": len(fake_users),
                "normal_users": len(normal_users),
                "detected_users": len(detected_users),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TPR": tpr,
                "FPR": fpr,
            })

    result_df = pd.DataFrame(results)

    output_path = (
        project_root / "day2_rule3_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("===== 실험 완료 =====")
    print(result_df.to_string(index=False))
    print()
    print(f"결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()