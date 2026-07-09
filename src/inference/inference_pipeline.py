#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import json
from datetime import datetime
from typing import List
from pathlib import Path

import hydra
from hydra import compose
from hydra.utils import instantiate

from src.utils.helpers import prepare_metadata, add_think_tag


def load_input_data_and_language(template_file: str, limit: int = None):
    data = json.load(open(template_file, "r"))["sampled_data"]
    if limit:
        data = data[:limit]
    language = data[0]["metadata"]["language"]
    return data, language


def process_prompts(
    model, input_data: List[dict], think_mode: bool | None = None
) -> List[dict]:
    prompts = [x["prompt"] for x in input_data]
    if think_mode is not None:
        prompts = [add_think_tag(p, think_mode) for p in prompts]
    responses = model.evaluate(prompts)
    return [
        {"id": dp["id"], "prediction": resp} for dp, resp in zip(input_data, responses)
    ]


def save_results(input_file: str, metadata: dict, results: List[dict]):
    inf_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gen_file = Path(input_file)
    gen_ts = gen_file.parent.name
    gen_path = str(gen_file)

    # Build results root by swapping out 'generated' for 'results'
    parts = list(gen_file.parts)
    idx = parts.index("generated")
    results_root = Path(*parts[:idx]) / "results"
    tail = Path(*parts[idx + 1 : -1])

    model_name = metadata["model"]["model_name"]
    prompt_method = metadata["system_method"]
    think_mode = metadata["model"]["think_mode"]
    if think_mode is not None:
        prompt_method = prompt_method + f"_think_mode={think_mode}"
    out_dir = results_root / tail / model_name / prompt_method / inf_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / "results.json"
    payload = {
        "generation_timestamp": gen_ts,
        "generation_path": gen_path,
        "inference_timestamp": inf_ts,
        "model": model_name,
        "hydra_config": metadata,
        "results": results,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    print(f"✔️  Results written to {result_file}")


def process_one_file(cfg, limit):
    template_file = cfg.filename

    input_data, language = load_input_data_and_language(template_file, limit)
    cfg.instruction_prefix = cfg.instruction_prefix.format(language=language)
    sys_cfg = compose(
        config_name=f"system_instructions/{cfg.scenario.value}/{cfg.system_method}"
    )
    cfg.system_instructions = sys_cfg["system_instructions"][cfg.scenario.value][
        "value"
    ]
    cfg.model.system_instructions = (
        f"{cfg.instruction_prefix} {cfg.system_instructions}"
    )
    model = instantiate(cfg.model)
    think_mode = cfg.model.get("think_mode", None)
    results = process_prompts(model, input_data, think_mode)
    metadata = prepare_metadata(cfg)
    save_results(template_file, metadata, results)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    process_one_file(cfg)


if __name__ == "__main__":
    main()
