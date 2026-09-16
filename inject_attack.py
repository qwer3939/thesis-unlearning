#!/usr/bin/env python3
# Usage: python inject_attack.py --input ml-1m/ratings.dat --attack-ratio 0.03
# Usage: python inject_attack.py --input ml-1m/ratings.dat --attack-ratio all
# Usage: python inject_attack.py --input ../ml-1m/ratings.dat --output-dir ../attack_output
"""Inject bandwagon profiles with 15 popular and 15 other fillers by default."""

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
import random
import shutil


RATIOS = ("0.01", "0.03", "0.05")

# Change this value to adjust the total number of filler items per fake user.
# The script splits it as evenly as possible between popular and other items.
DEFAULT_N_FILLERS = 100


def scan_ratings(path):
    item_counts = Counter()
    user_ids = set()
    max_user_id = 0
    min_timestamp = None
    max_timestamp = None
    row_count = 0

    with path.open(encoding="ascii") as source:
        for row_count, line in enumerate(source, 1):
            fields = line.strip().split("::")
            if len(fields) != 4:
                raise ValueError(f"line {row_count}: expected four ::-separated fields")
            try:
                user_id, item_id, rating, timestamp = map(int, fields)
            except ValueError as exc:
                raise ValueError(f"line {row_count}: fields must be integers") from exc
            if not 1 <= rating <= 5:
                raise ValueError(f"line {row_count}: rating must be 1 through 5")

            user_ids.add(user_id)
            item_counts[item_id] += 1
            max_user_id = max(max_user_id, user_id)
            min_timestamp = (
                timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            )
            max_timestamp = (
                timestamp if max_timestamp is None else max(max_timestamp, timestamp)
            )

    if not row_count:
        raise ValueError("original ratings file is empty")
    return (
        row_count, len(user_ids), max_user_id, item_counts,
        min_timestamp, max_timestamp,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ml-1m/ratings.dat"))
    parser.add_argument("--output-dir", type=Path, default=Path("attack_output"))
    parser.add_argument("--attack-ratio", choices=(*RATIOS, "all"), default="all")
    parser.add_argument("--n-targets", type=int, default=1)
    parser.add_argument("--n-fillers", type=int, default=DEFAULT_N_FILLERS)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"ratings file not found: {args.input}")
    if args.n_targets < 1 or args.n_fillers < 0:
        parser.error("n_targets must be >= 1 and n_fillers must be >= 0")

    original_rows, user_count, max_id, counts, time_min, time_max = (
        scan_ratings(args.input)
    )
    rng = random.Random(args.seed)

    # The original task does not define 'unpopular'; use the lowest 20% of
    # original item counts while keeping at least 10 real ratings per target.
    popular = sorted(counts, key=lambda item: (-counts[item], item))[:100]
    popular_set = set(popular)
    unpopular = sorted(counts, key=lambda item: (counts[item], item))
    bottom = unpopular[:ceil(len(unpopular) * 0.20)]
    candidates = [
        item for item in bottom if counts[item] >= 10 and item not in popular_set
    ]
    if len(candidates) < args.n_targets:
        parser.error("not enough unpopular target items with 10+ real ratings")
    targets = rng.sample(candidates, args.n_targets)

    # Other fillers are randomly selected from rated movies outside the top
    # 100. Exclude targets so each fake user rates each movie only once.
    other = sorted(item for item in counts if item not in popular_set and item not in targets)
    popular_fillers = args.n_fillers // 2
    other_fillers = args.n_fillers - popular_fillers
    if popular_fillers > len(popular) or other_fillers > len(other):
        parser.error("not enough distinct movies to sample the requested fillers")

    # Generate the 5% group once. Smaller ratios use identical prefixes,
    # isolating the number of fake users as the only difference.
    max_fake_count = round(user_count * 0.05)
    profiles = []
    for index in range(max_fake_count):
        fake_id = max_id + index + 1
        rows = []
        filler_ids = rng.sample(popular, popular_fillers) + rng.sample(other, other_fillers)
        for item_id in filler_ids:
            rating = rng.choice((4, 5))
            timestamp = rng.randint(time_min, time_max)
            rows.append(f"{fake_id}::{item_id}::{rating}::{timestamp}\n")
        for item_id in targets:
            timestamp = rng.randint(time_min, time_max)
            rows.append(f"{fake_id}::{item_id}::5::{timestamp}\n")
        profiles.append(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_items.txt").write_text(
        "".join(f"{item}\n" for item in targets), encoding="ascii"
    )

    selected = RATIOS if args.attack_ratio == "all" else (args.attack_ratio,)
    for ratio in selected:
        fake_count = round(user_count * float(ratio))
        attacked = args.output_dir / f"ratings_attacked_{ratio}.dat"
        if attacked.resolve() == args.input.resolve():
            parser.error("output would overwrite the original ratings file")
        fake_users = args.output_dir / f"fake_users_{ratio}.txt"

        with args.input.open("rb") as source, attacked.open("wb") as output:
            source.seek(-1, 2)
            ends_with_newline = source.read(1) == b"\n"
            source.seek(0)
            shutil.copyfileobj(source, output)
            if not ends_with_newline:
                output.write(b"\n")
            for profile in profiles[:fake_count]:
                for row in profile:
                    output.write(row.encode("ascii"))

        fake_users.write_text(
            "".join(f"{max_id + index + 1}\n" for index in range(fake_count)),
            encoding="ascii",
        )
        expected = original_rows + fake_count * (args.n_fillers + args.n_targets)
        print(f"{ratio}: {fake_count} fake users, {expected} expected total ratings")
        print(f"  {attacked} | {fake_users}")

    print(f"targets: {targets} -> {args.output_dir / 'target_items.txt'}")
    print(f"fillers per fake user: {popular_fillers} top-100 + {other_fillers} other")


if __name__ == "__main__":
    main()
