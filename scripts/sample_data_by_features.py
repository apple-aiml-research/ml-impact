#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from src.utils.io import generate_verification_csv


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_feature_combo(
    data,
    judge_scenario=False,
    exclude_fields={"base_word", "inflection", "name", "base_motion_verb"},
):
    feat_buckets = defaultdict(list)
    feat_values = defaultdict(lambda: defaultdict(set))
    for d in data:
        slot_features = d["metadata"]["slot_features"]
        combo_parts = []
        for slot, features in slot_features.items():
            combo_parts.append(slot)
            for feat, val in features.items():
                if feat not in exclude_fields:
                    combo_parts.append(str(val))
                    feat_values[slot][feat].add(val)
        if judge_scenario:
            combo_parts.append(d["gold"])
        key = "_".join(combo_parts)
        feat_buckets[key].append(d)

    # convert set to list for json dump
    for slot in feat_values:
        for feat in feat_values[slot]:
            feat_values[slot][feat] = list(feat_values[slot][feat])
    return feat_buckets, feat_values


def sample_buckets(buckets, n_per_combo=100, seed=42):
    sampled = []
    count = {}
    random.seed(seed)

    # if too much buckets we random sample less
    # examples from the no buckets
    if len(buckets) > 100:
        yes_buckets_keys = [x for x in buckets if "Yes" in x]
        num_yes_examples = sum([len(buckets[x]) for x in yes_buckets_keys])

        no_bucket_keys = [x for x in buckets if "No" in x]
        num_no_examples_per_bucket = num_yes_examples // len(no_bucket_keys)
        for key, value in buckets.items():
            if "No" in key:
                random.seed(42)
                buckets[key] = random.sample(
                    value, min(num_no_examples_per_bucket, len(value))
                )

    for combo_key, items in buckets.items():
        k = min(n_per_combo, len(items))
        count[combo_key] = k
        samples = random.sample(items, k)

        # make sure we do not have duplicate prompts
        prompts = [s["prompt"] for s in samples]
        while len(set(prompts)) != k:
            seed += 1
            random.seed(seed)
            samples = random.sample(items, k)
            prompts = [s["prompt"] for s in samples]
        for s in samples:
            s["metadata"]["bucket_key"] = combo_key
        sampled.extend(samples)

    return sampled, count


def save_sampled_data(
    output_path, sampled_data, count, feat_values, verbose: bool = False
):
    feat_values = dict(feat_values)
    num_buckets = len(list(count.keys()))
    num_prompts = sum(list(count.values()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sampled_data": sampled_data,
                "feature_values": feat_values,
                "count": {
                    "per_bucket": count,
                    "num_buckets": num_buckets,
                    "num_prompts": num_prompts,
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
    if verbose:
        print(f"Sampled data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample examples based on slot features."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save sampled JSON output."
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of samples per combination."
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Sample Seed",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Whether to output csv verification files or not.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print statements",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"sampled_{input_path.name}"
    )

    data = load_data(input_path)
    judge_scenario = data[0]["metadata"].get("scenario") == "judge"
    feat_buckets, feat_values = group_by_feature_combo(data, judge_scenario)
    sampled_data, count = sample_buckets(
        feat_buckets, n_per_combo=args.n, seed=args.sample_seed
    )
    save_sampled_data(output_path, sampled_data, count, feat_values, args.verbose)

    if args.csv:
        generate_verification_csv(output_path, args.verbose)


if __name__ == "__main__":
    main()
