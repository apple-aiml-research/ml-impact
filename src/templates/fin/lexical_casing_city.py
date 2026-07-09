#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pandas as pd
from typing import List, Dict, Tuple

from src.utils.text import str2list
from src.templates.template_base import TemplateBase, register_template


@register_template
class FinLexicalCasingCity(TemplateBase):
    """Generates data for lexical casing regarding cities"""

    def __init__(self):
        self.setup_common(template_name="lexical_casing_city.j2", lang_code="fin")

        self.features = [""]
        self.additional_features = {}
        self.mismatched_features = ["case"]
        # Cache correct city case
        self.correct_cities_df = self.placeholders.get_by_features("cities_correct")
        self.correct_cities_dict = dict(
            zip(self.correct_cities_df["base_word"], self.correct_cities_df["case"])
        )

    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        example_id = 1

        for base_city in self.correct_cities_dict:
            correct_case = self.correct_cities_dict[base_city]
            noun_feature_filter = {"base_word": base_city, "case": correct_case}

            correct_inflected_noun = self.get_inflected_forms(
                "cities_correct", noun_feature_filter
            )

            correct_sentence = self.sentence_template.render(
                scenario="generation"
            ).strip()
            correct_prompt = self.generation_template.render(
                language=self.language,
                pos=base_city,
                pos_name="destination",
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{base_city}_{example_id}_gen",
                "prompt": correct_prompt,
                "gold": correct_inflected_noun,
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "generation",
                    "template_name": self.template_name,
                    "sentence": correct_sentence,
                    "slot_features": {
                        "noun": noun_feature_filter,
                    },
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples

    def _generate_judgment(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        example_id = 1

        for base_city in self.correct_cities_dict:
            correct_case = self.correct_cities_dict[base_city]
            (
                correct_examples,
                correct_inflections,
                example_id,
            ) = self._create_correct_judgments(
                base_city,
                additional_instruction,
                example_id,
            )
            examples.extend(correct_examples)

            exclude_features = {
                "case": [correct_case],
            }

            (
                incorrect_examples,
                example_id,
            ) = self._create_incorrect_judgments(
                base_city,
                correct_inflections,
                additional_instruction,
                example_id,
                exclude_features,
            )
            examples.extend(incorrect_examples)

        return examples

    def _create_correct_judgments(
        self,
        base_noun: str,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], List[str], int]:
        """
        Generate correct judgment examples for lexical casing.

        Args:
            base_noun (str): base noun
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], List[str], Dict, int]: Correct examples, correct inflections, feature filter, and updated example ID.
        """

        examples = []

        noun_feature_filt = {
            "base_word": base_noun,
            "case": self.correct_cities_dict[base_noun],
        }

        correct_inflections = self.get_inflected_forms(
            "cities_correct", noun_feature_filt
        )

        for correct_inflection in correct_inflections:
            correct_sentence = self.sentence_template.render(
                noun=correct_inflection,
                scenario="judge",
            ).strip()
            correct_prompt = self.judge_template.render(
                language=self.language,
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{base_noun}_{example_id}_pos",
                "prompt": correct_prompt,
                "gold": "Yes",
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "judge",
                    "template_name": self.template_name,
                    "sentence": correct_sentence,
                    "slot_features": {
                        "noun": noun_feature_filt,
                    },
                    "correct_inflections": correct_inflections,
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples, correct_inflections, example_id

    def _create_incorrect_judgments(
        self,
        base_noun: str,
        correct_inflections: List[str],
        additional_instruction: str,
        example_id: int,
        exclude_features: Dict = None,
    ) -> Tuple[List[Dict], int]:
        """
        Generate incorrect judgment examples for lexical casing for cities.

        Args:
            base_noun (str): base noun
            correct_inflections (List[str]): List of correct number inflections.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.
            exclude_features (Dict, optional): Features to exclude

        Returns:
            Tuple[List[Dict], int]: Incorrect examples and updated example ID.
        """

        examples = []

        examples = []
        noun_df = self.get_inflected_forms(
            "cities",
            {"base_word": base_noun},
            return_df=True,
        )
        correct_inflection_set = set(correct_inflections)
        # We now have the correct inflection set and need to get the incorrect one
        # We can do this by filtering out the inflections that are different
        # But before, we might have scenarios where a different inflection is actually correct
        exclude_condition = pd.Series([True] * len(noun_df), index=noun_df.index)

        conditions = []
        if exclude_features:
            for feat, value in exclude_features.items():
                if feat not in noun_df.columns:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    conditions.append(noun_df[feat].isin(value))
                else:
                    conditions.append(noun_df[feat] == value)

        if conditions:
            exclude_condition = ~pd.concat(conditions, axis=1).all(axis=1)

        incorrect_noun_data = noun_df[
            exclude_condition
            & ~noun_df["inflection"].apply(
                lambda x: any(v in correct_inflection_set for v in str2list(x))
            )
        ]

        if incorrect_noun_data.empty:
            raise ValueError(
                f"There are no left inflected forms for {base_noun} with {correct_inflection_set}."
            )
        incorrect_noun_data = incorrect_noun_data.drop_duplicates(
            "inflection"
        )  # rm incorrect data having same inflections

        for index, incorrect_noun_row in incorrect_noun_data.iterrows():
            incorrect_inflections = str2list(incorrect_noun_row["inflection"])
            for incorrect_inflection in incorrect_inflections:
                incorrect_sentence = self.sentence_template.render(
                    noun=incorrect_inflection,
                    scenario="judge",
                ).strip()
                incorrect_prompt = self.judge_template.render(
                    language=self.language,
                    sentence=incorrect_sentence,
                    additional_instruction=additional_instruction,
                )

                sentence_data = {
                    "id": f"{base_noun}_{example_id}_neg",
                    "prompt": incorrect_prompt,
                    "gold": "No",
                    "metadata": {
                        "language": self.language,
                        "lang_code": self.lang_code,
                        "scenario": "judge",
                        "template_name": self.template_name,
                        "sentence": incorrect_sentence,
                        "slot_features": {
                            "noun": {
                                "base_word": base_noun,
                                "case": incorrect_noun_row["case"],
                            },
                        },
                        "correct_inflections": correct_inflections,
                        "mismatched_features": self.mismatched_features,
                    },
                }
                examples.append(sentence_data)
                example_id += 1
        return examples, example_id
