#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import re
import json
import random
import pandas as pd
import plotly.express as px
from pathlib import Path

import matplotlib.colors as mcolors
from omegaconf import OmegaConf, DictConfig

from src.config import OUTPUT_PATH
from src.utils.text import post_process_response


def is_eq(x: str, y: str) -> bool:
    return x == y


def is_intersection(x: list, y: list) -> bool:
    x = [xx.lower() for xx in x]
    y = [yy.lower() for yy in y]
    return bool(set(x).intersection(y))


def prepare_metadata(model_cfg: DictConfig):
    metadata = OmegaConf.to_container(model_cfg, resolve=True)
    metadata["model"].pop("project", None)
    return metadata


def validate_judge_answer(response: str):
    isYes = bool(re.search(r"\byes\b", response.lower()))
    isNo = bool(re.search(r"\bno\b", response.lower()))
    isValid = isYes ^ isNo
    if not isValid:
        answer = random.choice(["Yes", "No"])
    else:
        answer = "Yes" if isYes else "No"
    return answer, isValid


def add_think_tag(prompt: str, think_mode: bool):
    tag = "/no_think" if not think_mode else "/think"
    prompt = prompt + tag
    return prompt


def analyze_result_file(results_file: Path):
    """
    Given one JSON results_file, returns:
      1) df_all       : one row per prediction
      2) summary_df   : per-bucket summary (+ ALL row)
      3) mismatch_df  : only rows where valid==False or correct==False
    """

    results_data = json.load(open(results_file, "r"))
    input_fn = results_data["generation_path"]
    input_data = json.load(open(input_fn, "r"))
    predictions = results_data["results"]
    input_sample_data = input_data["sampled_data"]
    scenario = input_sample_data[0]["metadata"]["scenario"]
    lang = input_sample_data[0]["metadata"]["lang_code"]
    model_name = results_data["hydra_config"]["model"]["model_name"]
    prompting_method = results_data["hydra_config"]["system_method"]

    gold_dict = {entry["id"]: entry["gold"] for entry in input_sample_data}
    meta_dict = {
        entry["id"]: (
            entry["metadata"]["bucket_key"],
            entry.get("prompt", ""),
            "_".join(entry["metadata"].get("mismatched_features", [])),
        )
        for entry in input_sample_data
    }

    records = []
    for pred in predictions:
        pred_id = pred["id"]
        if pred_id not in gold_dict:
            continue

        raw_ans = pred["prediction"]["response"]
        postprocessed_answer = post_process_response(raw_ans, lang)
        gold_label = gold_dict[pred_id]

        if scenario == "judge":
            postprocessed_answer_to_compare, isValid = validate_judge_answer(
                postprocessed_answer
            )
            eval_fn = is_eq
        elif scenario == "generation":
            postprocessed_answer_to_compare = postprocessed_answer.split(" ")
            gold_label = [post_process_response(label, lang) for label in gold_label]
            eval_fn = is_intersection
            isValid = True if postprocessed_answer_to_compare else False

        else:
            raise ValueError("Wrong Scenario.")

        isCorrect = eval_fn(postprocessed_answer_to_compare, gold_label)
        bucket_key, prompt, mismatched_features = meta_dict.get(pred_id, (None, ""))

        records.append(
            {
                "file": str(results_file),
                "scenario": scenario,
                "lang": lang,
                "mismatched_features": mismatched_features,
                "model_name": model_name,
                "prompting_method": prompting_method,
                "bucket_key": bucket_key,
                "id": pred_id,
                "prompt": prompt,
                "raw_pred": raw_ans,
                "post_pred": postprocessed_answer,
                "gold": gold_label,
                "valid": isValid,
                "correct": isCorrect,
            }
        )

    df_all = pd.DataFrame(records)

    # 1) Per‑bucket summary
    summary_df = (
        df_all.groupby(["bucket_key", "mismatched_features"])
        .agg(
            n_total=("id", "count"),
            n_correct=("correct", "sum"),
            n_valid=("valid", "sum"),
        )
        .assign(
            n_incorrect=lambda d: d.n_total - d.n_correct,
            n_invalid=lambda d: d.n_total - d.n_valid,
            accuracy=lambda d: d.n_correct / d.n_total,
            validity=lambda d: d.n_valid / d.n_total,
        )
        .reset_index()
    )
    # 2) Global summary row
    global_rows = []
    if scenario == "judge":
        df_all_yes = df_all[df_all.bucket_key.str.contains("_Yes")]
        global_row_yes = get_global_statistics(df_all_yes, "ALL_YES")

        df_all_no = df_all[df_all.bucket_key.str.contains("_No")]
        global_row_no = get_global_statistics(df_all_no, "ALL_NO")

        global_rows.extend([global_row_yes, global_row_no])

    global_row = get_global_statistics(df_all, "ALL")
    global_rows.append(global_row)

    summary_df = pd.concat([summary_df, pd.DataFrame(global_rows)], ignore_index=True)
    summary_df["scenario"] = scenario

    file_parts = results_file.parts
    summary_df["template"] = file_parts[file_parts.index(lang) + 1]

    # 3) Mismatches table
    mismatch_df = df_all.loc[
        ~df_all["correct"] | ~df_all["valid"],
        [
            "bucket_key",
            "id",
            "prompt",
            "raw_pred",
            "post_pred",
            "gold",
            "valid",
            "correct",
            "model_name",
            "prompting_method",
            "scenario",
            "mismatched_features",
        ],
    ]

    return df_all, summary_df, mismatch_df


def get_global_statistics(df: pd.DataFrame, global_label: str = "ALL") -> dict:
    global_row = {
        "bucket_key": global_label,
        "n_total": df.shape[0],
        "n_correct": df["correct"].sum(),
        "n_valid": df["valid"].sum(),
    }
    global_row.update(
        {
            "n_incorrect": global_row["n_total"] - global_row["n_correct"],
            "n_invalid": global_row["n_total"] - global_row["n_valid"],
            "accuracy": global_row["n_correct"] / global_row["n_total"]
            if global_row["n_total"]
            else "",
            "validity": global_row["n_valid"] / global_row["n_total"]
            if global_row["n_total"]
            else "",
        }
    )
    return global_row


def highlight_accuracy_pos_neg(val):
    """
    Green = good (>0)
    Orange = mid (0)
    Red = low (<0)
    """
    if val > 0:
        return "background-color: #006400; color: white"  # DarkGreen
    elif val < 0:
        return "background-color: #8B0000; color: white"  # DarkRed
    else:
        return "background-color: #8B6508; color: white"  # DarkGoldenRod


def highlight_gradient(val):
    if pd.isna(val):
        return ""
    try:
        val = float(val)
    except (ValueError, TypeError):
        return ""

    val = max(0, min(1, val))

    # Hue from 0 (red) to 120 (green)
    hue = val * 120 / 360  # normalize hue to [0,1] for hsv_to_rgb

    saturation = 0.99  # 70% saturation for less bright colors
    value = 0.6  # 50% brightness for darker colors

    r, g, b = mcolors.hsv_to_rgb([hue, saturation, value])
    r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return f"background-color: rgb({r}, {g}, {b}); color: white"


def get_template_gen_vs_judge(
    lang: str,
    template_name: str,
    model_name: str,
    prompting_method: str,
    output_files: bool = True,
) -> pd.DataFrame:
    df_gen_judge, cols = [], []
    statistics = []
    result_dir = Path(OUTPUT_PATH)
    result_path = result_dir / lang / template_name
    for scenario in ["generation", "judge"]:
        result_file = list(
            Path(result_path).rglob(
                f"**/{scenario}/**/{model_name}/**/{prompting_method}/**/*.json"
            )
        )[0]

        df, summ, mm = analyze_result_file(result_file)
        output_path = result_file.parent
        if output_files:
            df.to_csv(output_path / "detailed.csv", index=False)
            summ.to_csv(output_path / "summary.csv", index=False)
            mm.to_csv(output_path / "mismatched.csv", index=False)

        if scenario == "generation":
            gen_all_row = summ[summ.bucket_key == "ALL"].squeeze()
            statistics.extend(
                [
                    gen_all_row["accuracy"],
                    f'{gen_all_row['n_correct']}/{gen_all_row['n_total']}',
                ]
            )
            cols.extend(["Gen_Acc", "Gen_Ratio"])

        elif scenario == "judge":
            gen_all_yes_row = summ[summ.bucket_key == "ALL_YES"].squeeze()
            statistics.extend(
                [
                    gen_all_yes_row["accuracy"],
                    f'{gen_all_yes_row['n_correct']}/{gen_all_yes_row['n_total']}',
                ]
            )
            cols.extend(["Judge_Yes_Acc", "Judge_Yes_Ratio"])

            gen_all_no_row = summ[summ.bucket_key == "ALL_NO"].squeeze()
            statistics.extend(
                [
                    gen_all_no_row["accuracy"],
                    f'{gen_all_no_row['n_correct']}/{gen_all_no_row['n_total']}',
                ]
            )
            cols.extend(["Judge_No_Acc", "Judge_No_Ratio"])

    df_gen_judge.append([lang, template_name] + statistics)
    cols = ["Language", "Template"] + cols
    df_gen_judge = pd.DataFrame(df_gen_judge, columns=cols)
    return df_gen_judge


def get_lang_templates_gen_vs_judge(
    lang: str, model_name: str, prompting_method: str, output_files: bool = True
) -> pd.DataFrame:
    df_gen_judge_lang = []
    result_dir = Path(OUTPUT_PATH)
    result_path = result_dir / lang

    for template_path in result_path.glob("*"):
        if not template_path.is_dir():
            continue
        template_name = template_path.name
        df_gen_judge = get_template_gen_vs_judge(
            lang, template_name, model_name, prompting_method, output_files
        )
        df_gen_judge_lang.append(df_gen_judge)

    df_gen_judge_lang = pd.concat(df_gen_judge_lang).reset_index(drop=True)
    return df_gen_judge_lang


def ratio_sum(series):
    total_num = 0
    total_denom = 0
    for val in series:
        num, denom = map(int, val.split("/"))
        total_num += num
        total_denom += denom
    return f"{total_num}/{total_denom}"


def reshape_to_long_format(df, x_column):
    rows = []
    for _, row in df.iterrows():
        x_column_value = row[x_column]
        for scenario, acc_col, ratio_col in [
            ("Generation", "Gen_Acc", "Gen_Ratio"),
            ("Judge-Yes", "Judge_Yes_Acc", "Judge_Yes_Ratio"),
            ("Judge-No", "Judge_No_Acc", "Judge_No_Ratio"),
        ]:
            rows.append(
                {
                    x_column: x_column_value,
                    "Scenario": scenario,
                    "Accuracy": row[acc_col],
                    "Ratio": row[ratio_col],
                }
            )

    return pd.DataFrame(rows)


def plot_histogram(df, x_axis_label, y_axis_label="Accuracy", color="Scenario"):
    fig = px.histogram(
        reshape_to_long_format(df, x_axis_label),
        x=x_axis_label,
        y=y_axis_label,
        color=color,
        barmode="group",
        height=400,
    )
    return fig


def get_result_dict(
    models,
    prompting_method,
    RESULT_DIR,
    languages=["eng", "ara", "heb", "fin", "rus", "tur"],
    threshold=2,
    output_files=False,
):
    "Compile results into a dictionary for visualization."
    result_dict = {}
    for model_name in models:
        result_dict[model_name] = {}
        result_dict[model_name][prompting_method] = {}
        for lang in languages:
            lang_dir = RESULT_DIR / lang
            result_dict[model_name][prompting_method][lang] = {}
            for template_dir in lang_dir.glob("*"):
                template_name = template_dir.name
                result_dict[model_name][prompting_method][lang][template_name] = {}
                for scenario_dir in template_dir.glob("*"):
                    scenario = scenario_dir.name
                    result_dict[model_name][prompting_method][lang][template_name][
                        scenario
                    ] = []
                    data_gen_ts_dirs = list(scenario_dir.glob("*"))
                    assert len(data_gen_ts_dirs) == 1, f"Check {scenario_dir}"
                    data_gen_ts_dir = data_gen_ts_dirs[0]
                    inference_ts_dirs = list(
                        data_gen_ts_dir.glob(f"{model_name}/{prompting_method}*/*")
                    )
                    inference_ts_dirs = [x for x in inference_ts_dirs if x.is_dir()]
                    assert (
                        len(inference_ts_dirs) == 1
                    ), f"The number of directories is {len(inference_ts_dirs)} when it should be one. Check {data_gen_ts_dir} with {model_name}/{prompting_method}"
                    if len(inference_ts_dirs) != 1:
                        print(
                            f"The number of directories is {len(inference_ts_dirs)} when it should be one. Check {data_gen_ts_dir} with {model_name}/{prompting_method}"
                        )
                        print(
                            f"Run the command:\npython scripts/run_inference.py  --file {data_gen_ts_dir}/sampled_data.json --scenario {'generation' if 'generation' in str(data_gen_ts_dir) else 'judge'} system_method={prompting_method} model={model_name}"
                        )
                        continue
                    inference_ts_dir = inference_ts_dirs[0]
                    result_file = inference_ts_dir / "results.json"
                    df, summary_df, mm = analyze_result_file(Path(result_file))
                    output_path = result_file.parent
                    if output_files:
                        df.to_csv(output_path / "detailed.csv", index=False)
                        summary_df.to_csv(output_path / "summary.csv", index=False)
                        mm.to_csv(output_path / "mismatched.csv", index=False)

                    summary_df_unique = summary_df[
                        ~summary_df.bucket_key.str.contains("ALL")
                    ]
                    summary_df_unique_filtered = summary_df_unique[
                        summary_df_unique["accuracy"] < threshold
                    ]
                    summary_df_unique_filtered = summary_df_unique_filtered[
                        ["bucket_key", "accuracy"]
                    ]
                    if not summary_df_unique_filtered.empty:
                        result_dict[model_name][prompting_method][lang][template_name][
                            scenario
                        ] = summary_df_unique_filtered.to_dict(orient="records")
    return result_dict
