#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
import pandas as pd


def safe_to_csv(df, filename, overwrite=False, **kwargs):
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(
            f"{filename} already exists. Set `overwrite=True` to overwrite it."
        )
    df.to_csv(filename, **kwargs)


def load_prompt_template(name: str, env):
    try:
        return env.get_template(name)
    except Exception as e:
        raise ValueError(f"Failed to load prompt template {name}: {e}")


def generate_verification_csv(
    filename: str, verbose: bool = False, num_samples: int = 10, seed: int = 42
):
    """This function takes the sampled generated data and outputs a csv to
    verify that the sentences."""
    filename = str(filename)
    with open(filename, "r") as f:
        data = json.load(f)

    if "sampled_data" not in data:
        raise ValueError("Please pass the sampled data!")

    sampled_data = data["sampled_data"]

    if not len(sampled_data):
        raise ValueError("No data to process!")

    scenario = sampled_data[0]["metadata"]["scenario"]

    gold_sentences = []

    # generation
    if scenario == "generation":
        for dp in sampled_data:
            prefix_sentence = dp["metadata"]["sentence"]
            gold = dp["gold"][0]
            gold_sentence = prefix_sentence.replace("_____", gold).strip()
            bucket_key = dp["metadata"]["bucket_key"]
            gold_sentences.append([gold_sentence, bucket_key])

        df_label = pd.DataFrame(gold_sentences, columns=["sentence", "bucket_key"])

    # judge
    elif scenario == "judge":
        gold_sentences = []
        for dp in sampled_data:
            sentence = dp["metadata"]["sentence"]
            label = dp["gold"]
            gold = dp["metadata"]["correct_inflections"]
            bucket_key = dp["metadata"]["bucket_key"]
            gold_sentences.append([sentence, label, gold, bucket_key])

        df_label = pd.DataFrame(
            gold_sentences,
            columns=["sentence", "label", "correct_inflection", "bucket_key"],
        )
        df_label = df_label.sort_values(by="label", ascending=False)

    else:
        raise ValueError(f"Unknown scenario type: {scenario}")

    out_fn = filename.replace(".json", ".csv")
    sampled_df_label = (
        df_label.groupby("bucket_key")
        .apply(lambda x: x.sample(n=min(num_samples, len(x)), random_state=seed))
        .reset_index(drop=True)
        .drop(columns=["bucket_key"])
    )
    sampled_df_label.to_csv(out_fn, index=False)
    if verbose:
        print(f"Verification csv saved to {out_fn}")
