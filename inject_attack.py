#!/usr/bin/env python3
# Usage: python inject_attack.py --input ml-1m/ratings.dat --attack-ratio 0.03
# Usage: python inject_attack.py --input ml-1m/ratings.dat --attack-ratio all
# Usage: python inject_attack.py --input ../ml-1m/ratings.dat --output-dir ../attack_output

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
import random
import shutil

RATIOS = ("0.01", "0.03", "0.05")


def scan_ratings(path):
    item_counts = Counter()
    user_ids = set()
    max_user_id = 0
    time_min = None
    time_max = None
    row_count = 0

    with path.open(encoding="ascii") as source:
        for row_count, line in enumerate(source, 1):
            parts = line.strip().split("::")
            if len(parts) != 4:
                raise ValueError(f"{row_count}행: 열이 4개가 아닙니다")
            user, item, rating, timestamp = map(int, parts)
            if not 1 <= rating <= 5:
                raise ValueError(f"{row_count}행: 평점이 1~5 범위 밖입니다")

            user_ids.add(user)
            item_counts[item] += 1
            max_user_id = max(max_user_id, user)
            time_min = timestamp if time_min is None else min(time_min, timestamp)
            time_max = timestamp if time_max is None else max(time_max, timestamp)

    if row_count == 0:
        raise ValueError("원본 평점 파일이 비어 있습니다")

    return row_count, len(user_ids), max_user_id, item_counts, time_min, time_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("ml-1m/ratings.dat"))
    parser.add_argument("--output-dir", type=Path, default=Path("attack_output"))
    parser.add_argument("--attack-ratio", choices=(*RATIOS, "all"), default="all")
    parser.add_argument("--n-targets", type=int, default=1)
    parser.add_argument("--n-fillers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"원본 파일이 없습니다: {args.input}")
    if args.n_targets < 1 or not 0 <= args.n_fillers <= 100:
        parser.error("n_targets는 1 이상, n_fillers는 0~100이어야 합니다")

    original_rows, user_count, max_id, counts, time_min, time_max = (
        scan_ratings(args.input)
    )
    rng = random.Random(args.seed)

    # 원본 평점 수만 사용해 인기/비인기 아이템을 선정한다.
    popular = sorted(counts, key=lambda item: (-counts[item], item))[:100]
    unpopular = sorted(counts, key=lambda item: (counts[item], item))
    bottom_20_percent = unpopular[:ceil(len(unpopular) * 0.20)]
    candidates = [
        item for item in bottom_20_percent
        if counts[item] >= 10 and item not in popular
    ]
    if len(candidates) < args.n_targets:
        parser.error("조건에 맞는 비인기 타깃 영화가 부족합니다")

    targets = rng.sample(candidates, args.n_targets)

    # 5% 인원까지 한 번 생성한다. 1%, 3%는 이 목록의 앞부분만 사용한다.
    max_fake_count = round(user_count * 0.05)
    profiles = []
    for index in range(max_fake_count):
        fake_id = max_id + index + 1
        rows = []

        for item in rng.sample(popular, args.n_fillers):
            rating = rng.choice((4, 5))
            timestamp = rng.randint(time_min, time_max)
            rows.append(f"{fake_id}::{item}::{rating}::{timestamp}\n")

        for item in targets:
            timestamp = rng.randint(time_min, time_max)
            rows.append(f"{fake_id}::{item}::5::{timestamp}\n")

        profiles.append(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_items.txt").write_text(
        "".join(f"{item}\n" for item in targets), encoding="ascii"
    )

    selected_ratios = RATIOS if args.attack_ratio == "all" else (args.attack_ratio,)
    for ratio in selected_ratios:
        fake_count = round(user_count * float(ratio))
        attacked_path = args.output_dir / f"ratings_attacked_{ratio}.dat"
        fake_users_path = args.output_dir / f"fake_users_{ratio}.txt"

        with args.input.open("rb") as source, attacked_path.open("wb") as output:
            source.seek(-1, 2)
            ends_with_newline = source.read(1) == b"\n"
            source.seek(0)
            shutil.copyfileobj(source, output)
            if not ends_with_newline:
                output.write(b"\n")

            for profile in profiles[:fake_count]:
                for row in profile:
                    output.write(row.encode("ascii"))

        fake_users_path.write_text(
            "".join(f"{max_id + index + 1}\n" for index in range(fake_count)),
            encoding="ascii",
        )

        expected_rows = original_rows + fake_count * (
            args.n_fillers + args.n_targets
        )
        print(
            f"{ratio}: 가짜 사용자 {fake_count}명, "
            f"예상 평점 {expected_rows}행 → {attacked_path}"
        )

    print(f"타깃: {targets} → {args.output_dir / 'target_items.txt'}")


if __name__ == "__main__":
    main()