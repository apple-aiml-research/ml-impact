#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import json
import traceback
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

from src.config import CODE2LANG
from src.templates.template_base import TEMPLATE_REGISTRY


def generate_data_from_template(
    template_name: str, scenario: str, additional_instruction: str = ""
) -> Tuple[List[Dict], str]:
    """
    Generates data using the provided template and scenario.

    template_name: The name of the template to use for generation.
    scenario: The scenario (e.g., "generation", "judge") for which to generate data.
    :additional_instruction: Additional instructions, if any.
    return A list of generated examples.
    """
    # Retrieve the template class from the registry
    template_cls = TEMPLATE_REGISTRY.get(template_name)

    if not template_cls:
        raise ValueError(f"Template '{template_name}' not found in the registry.")

    # Instantiate the template
    template_instance = template_cls()

    # Generate the data based on the scenario
    examples = template_instance.generate(
        scenario, additional_instruction=additional_instruction
    )

    # get_output_path
    output_path = template_instance.get_output_path()

    return examples, output_path


def save_examples_to_file(examples, output_file, verbose=False):
    """Save the generated examples to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)
    if verbose:
        print(f"Data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate data from a template")

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument(
        "--template",
        type=str,
        help="The template name to use from the TEMPLATE_REGISTRY",
    )
    group1.add_argument(
        "--all_templates",
        action="store_true",
        help="Run for every template in TEMPLATE_REGISTRY",
    )
    group1.add_argument(
        "--lang_code",
        choices=list(CODE2LANG.keys()),
        help="Only run templates for this language (e.g., 'ara', 'eng').",
    )

    # Scenario (either "generation" or "judge")
    group2 = parser.add_mutually_exclusive_group(required=True)

    group2.add_argument(
        "--scenario",
        type=str,
        choices=["generation", "judge"],
        help="The scenario for data generation.",
    )
    group2.add_argument(
        "--all_scenarios", action="store_true", help="Runs for all scenarios"
    )

    # Additional instructions if needed
    parser.add_argument(
        "--additional_instruction",
        type=str,
        default="",
        help="Additional instructions to customize the generation.",
    )

    # Output file path
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generated",
        help="The output filepath to save the generated data (JSON format).",
    )

    parser.add_argument(
        "--sample", action="store_true", help="Run sampling after data generation."
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=100,
        help="Samples per feature combination if sampling.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Sample Seed",
    )
    parser.add_argument(
        "--sample_output_dir", type=str, help="Output dir for sampled data (optional)."
    )

    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Whether to output csv verification files or not.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print statements",
    )
    args = parser.parse_args()

    templates = (
        list(TEMPLATE_REGISTRY.keys()) if args.all_templates else [args.template]
    )
    if args.all_templates:
        templates = list(TEMPLATE_REGISTRY.keys())
    elif args.lang_code:
        templates = [
            k for (k, v) in TEMPLATE_REGISTRY.items() if v().lang_code == args.lang_code
        ]
    else:
        templates = [args.template]

    scenarios = ["generation", "judge"] if args.all_scenarios else [args.scenario]
    for template in templates:
        for scenario in scenarios:
            try:
                if args.verbose:
                    print(f"Running {scenario} scenario for the template {template}.")
                # Generate the data
                examples, output_path = generate_data_from_template(
                    template_name=template,
                    scenario=scenario,
                    additional_instruction=args.additional_instruction,
                )
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = Path(args.output_dir) / output_path / scenario / timestamp
                output_dir.mkdir(parents=True, exist_ok=True)
                output_fn = output_dir / "data.json"
                # Save the generated data to the specified output file
                save_examples_to_file(examples, output_fn, args.verbose)

                if args.sample:
                    import subprocess

                    sample_cmd = [
                        "python",
                        "scripts/sample_data_by_features.py",
                        "--input",
                        str(output_fn),
                        "--n",
                        str(args.sample_n),
                    ]
                    if args.sample_output_dir:
                        sample_cmd.extend(
                            [
                                "--output",
                                str(
                                    args.sample_output_dir,
                                    "--sample_seed",
                                    str(args.sample_seed),
                                ),
                            ]
                        )
                    if not args.no_csv:
                        sample_cmd.extend(["--csv"])
                    if args.verbose:
                        sample_cmd.extend(["--verbose"])
                    subprocess.run(sample_cmd)
                    if args.verbose:
                        print()
            except Exception as e:
                exc_type, exc_obj, tb = sys.exc_info()
                fname = tb.tb_frame.f_code.co_filename
                line_number = tb.tb_lineno
                print(f"⚠️  Skipping {template} due to error: {e}")
                print(f"📄 File: {fname} | 🧵 Line: {line_number}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
