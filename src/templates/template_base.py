#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import TEMPLATES_DIR, CODE2LANG, CONJUNCTIONS
from src.utils.io import load_prompt_template
from src.utils.placeholder import PlaceholderLoader

TEMPLATE_REGISTRY = {}


def register_template(cls):
    key = cls.__name__
    TEMPLATE_REGISTRY[key] = cls
    return cls


class TemplateBase(ABC):
    def setup_common(self, template_name: str, lang_code: str):
        self.template_name = template_name
        self.lang_code = lang_code
        self.language = CODE2LANG[self.lang_code]
        self.mismatched_features = []

        self.templates_dir = Path(TEMPLATES_DIR)
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir / self.lang_code),
            autoescape=select_autoescape(),
        )
        self.prompt_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(),
        )

        self.sentence_template = load_prompt_template(self.template_name, self.env)
        self.generation_template = load_prompt_template(
            "generation_prompt.j2", self.prompt_env
        )
        self.judge_template = load_prompt_template("judge_prompt.j2", self.prompt_env)

        self.placeholders = PlaceholderLoader(self.lang_code)
        self.get_inflected_forms = self.placeholders.get_inflected_forms
        self.conjunction = CONJUNCTIONS[self.lang_code]

    def generate(self, scenario: str, additional_instruction: str = "") -> List[Dict]:
        """Generates data for the "generation" or "judgment" scenarios with the option to pass additional instructions."""
        if scenario == "generation":
            return self._generate_generation(
                additional_instruction=additional_instruction
            )
        elif scenario == "judge":
            return self._generate_judgment(
                additional_instruction=additional_instruction
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    @abstractmethod
    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        """
        Generate sentence examples in the 'generation' scenario.

        Args:
            additional_instruction (str): Additional instructions to include in the prompt.

        Returns:
            List[Dict]: List of example dictionaries containing prompts, gold answers, and metadata.
        """
        pass

    @abstractmethod
    def _generate_judgment(self, additional_instruction: str = "") -> List[Dict]:
        """
        Generate sentence examples in the 'judgment' scenario.

        Args:
            additional_instruction (str): Additional instructions to include in the prompt.

        Returns:
            List[Dict]: List of example dictionaries containing prompts, gold answers, and metadata.
        """
        pass

    def get_output_path(self):
        return self.lang_code + "/" + self.template_name.replace(".j2", "")

    def _get_mismatched_features(self, row, gold_features: Dict) -> List[str]:
        """Comapres the feature dicts of an incorrect row inflection and that of the gold inflection
        to get the mismatching features"""
        mismatches = []
        for feat, gold_val in gold_features.items():
            if (
                feat in self.mismatched_features
                and row.get(feat)
                and row[feat] != gold_val
            ):
                mismatches.append(feat)
        return mismatches
