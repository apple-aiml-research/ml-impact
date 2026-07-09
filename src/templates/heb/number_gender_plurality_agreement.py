#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pandas as pd
from typing import List, Dict, Tuple

from src.utils.text import str2list
from src.templates.template_base import TemplateBase, register_template


@register_template
class HebNumberGenderPluralityAgreement(TemplateBase):
    """Generates data for number-noun agreement in Hebrew sentences, focusing on gender and plurality."""

    def __init__(self):
        self.setup_common(
            template_name="number_gender_plurality_agreement.j2", lang_code="heb"
        )

        self.features = ["gender"]
        self.additional_features = {"plurality": "PL"}
        self.mismatched_features = self.features

        self.noun_categories = ["occupations", "objects"]
        # Cache noun data
        self.noun_cache = {
            "MASC": self.placeholders.get_by_features(
                self.noun_categories, **{"gender": "MASC"}
            ),
            "FEM": self.placeholders.get_by_features(
                self.noun_categories, **{"gender": "FEM"}
            ),
        }

    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        feature_combinations, _ = self.placeholders.get_feature_combinations(
            self.features, "numbers"
        )
        example_id = 1

        base_numbers = self.placeholders.get_by_features(
            "numbers", **self.additional_features
        ).base_word.unique()

        for feature_combination in feature_combinations:
            noun_feature_dict = dict(zip(self.features, feature_combination))
            noun_feature_dict.update(self.additional_features)
            base_nouns = self.noun_cache[noun_feature_dict["gender"]].base_word.unique()
            for base_noun in base_nouns:
                noun_feature_filter = {**noun_feature_dict, "base_word": base_noun}
                correct_noun_list = self.get_inflected_forms(
                    self.noun_categories, noun_feature_filter
                )

                for correct_noun in correct_noun_list:
                    for base_number in base_numbers:
                        number_feature_filter = {
                            **noun_feature_dict,
                            "base_word": base_number,
                        }
                        number_feature_filter = self._apply_correct_inflection_logic(
                            number_feature_filter
                        )  # apply agreement logic

                        correct_inflected_number = self.get_inflected_forms(
                            "numbers", number_feature_filter
                        )

                        correct_sentence = self.sentence_template.render(
                            noun=correct_noun, scenario="generation"
                        ).strip()
                        correct_prompt = self.generation_template.render(
                            language=self.language,
                            pos=base_number,
                            pos_name="cardinal number",
                            sentence=correct_sentence,
                            additional_instruction=additional_instruction,
                        )

                        sentence_data = {
                            "id": f"{base_number}_{example_id}_gen",
                            "prompt": correct_prompt,
                            "gold": correct_inflected_number,
                            "metadata": {
                                "language": self.language,
                                "lang_code": self.lang_code,
                                "scenario": "generation",
                                "template_name": self.template_name,
                                "sentence": correct_sentence,
                                "slot_features": {
                                    "noun": noun_feature_filter,
                                    "number": number_feature_filter,
                                },
                            },
                        }

                        examples.append(sentence_data)
                        example_id += 1

        return examples

    def _generate_judgment(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        feature_combinations, _ = self.placeholders.get_feature_combinations(
            self.features, "numbers"
        )
        example_id = 1

        base_numbers = self.placeholders.get_by_features(
            "numbers", **self.additional_features
        ).base_word.unique()

        for feature_combination in feature_combinations:
            noun_feature_dict = dict(zip(self.features, feature_combination))
            noun_feature_dict.update(self.additional_features)
            base_nouns = self.noun_cache[noun_feature_dict["gender"]].base_word.unique()

            for base_noun in base_nouns:
                noun_feature_filter = {**noun_feature_dict, "base_word": base_noun}
                correct_noun_list = self.get_inflected_forms(
                    self.noun_categories, noun_feature_filter
                )

                for correct_noun in correct_noun_list:
                    noun_features_full = {
                        **noun_feature_filter,
                        "inflection": correct_noun,
                    }

                    for base_number in base_numbers:
                        (
                            correct_examples,
                            correct_inflections,
                            feature_filter,
                            example_id,
                        ) = self._create_correct_judgments(
                            noun_features_full,
                            base_number,
                            additional_instruction,
                            example_id,
                        )
                        examples.extend(correct_examples)

                        (
                            incorrect_examples,
                            example_id,
                        ) = self._create_incorrect_judgments(
                            noun_features_full,
                            base_number,
                            feature_filter,
                            correct_inflections,
                            additional_instruction,
                            example_id,
                        )
                        examples.extend(incorrect_examples)

        return examples

    def _create_correct_judgments(
        self,
        noun_features: Dict,
        base_number: str,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], List[str], Dict, int]:
        """
        Generate correct judgment examples for noun-number agreement.

        Args:
            noun_features (Dict): Noun features
            base_number (str): masculine form of the cardinal number.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], List[str], Dict, int]: Correct examples, correct inflections, feature filter, and updated example ID.
        """

        examples = []
        number_feature_filt = {
            k: v for (k, v) in noun_features.items() if k in self.features
        }
        number_feature_filt = {
            **number_feature_filt,
            **self.additional_features,
            "base_word": base_number,
        }
        number_feature_filt = self._apply_correct_inflection_logic(number_feature_filt)

        correct_inflections = self.get_inflected_forms("numbers", number_feature_filt)

        for correct_inflection in correct_inflections:
            correct_sentence = self.sentence_template.render(
                noun=noun_features["inflection"],
                number=correct_inflection,
                scenario="judge",
            ).strip()
            correct_prompt = self.judge_template.render(
                language=self.language,
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{base_number}_{example_id}_pos",
                "prompt": correct_prompt,
                "gold": "Yes",
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "judge",
                    "template_name": self.template_name,
                    "sentence": correct_sentence,
                    "slot_features": {
                        "number": {
                            **number_feature_filt,
                            "inflection": correct_inflection,
                        },
                        "noun": noun_features,
                    },
                    "correct_inflections": correct_inflections,
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples, correct_inflections, number_feature_filt, example_id

    def _create_incorrect_judgments(
        self,
        noun_features: Dict,
        base_number: str,
        number_features: Dict,
        correct_inflections: List[str],
        additional_instruction: str,
        example_id: int,
        exclude_features: Dict = None,
    ) -> Tuple[List[Dict], int]:
        """
        Generate incorrect judgment examples for number-noun agreement, excluding specified feature values or lists.

        Args:
            noun_features (Dict): Noun features.
            base_number (str): masculine form of the number.
            number_features (Dict): number features (gender, plurality,).
            correct_inflections (List[str]): List of correct number inflections.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.
            exclude_features (Dict, optional): Features to exclude (e.g., {'gender': ['MASC']}).

        Returns:
            Tuple[List[Dict], int]: Incorrect examples and updated example ID.
        """

        examples = []

        num_df = self.get_inflected_forms(
            "numbers",
            {"base_word": base_number, **self.additional_features},
            return_df=True,
        )

        correct_inflection_set = set(correct_inflections)
        # We now have the correct inflection set and need to get the incorrect one
        # We can do this by filtering out the inflections that are different
        # But before, we might have scenarios where a different inflection is actually correct

        exclude_condition = pd.Series([True] * len(num_df), index=num_df.index)

        conditions = []

        if exclude_features:
            for feat, value in exclude_features.items():
                if feat not in num_df.columns:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    conditions.append(num_df[feat].isin(value))
                else:
                    conditions.append(num_df[feat] == value)

        if conditions:
            exclude_condition = ~pd.concat(conditions, axis=1).all(axis=1)

        incorrect_number_data = num_df[
            exclude_condition
            & ~num_df["inflection"].apply(
                lambda x: any(v in correct_inflection_set for v in str2list(x))
            )
        ]

        if incorrect_number_data.empty:
            return examples, example_id

        incorrect_number_data = incorrect_number_data.drop_duplicates(
            "inflection"
        )  # rm incorrect data having same inflections

        for _, incorrect_number_record in incorrect_number_data.iterrows():
            incorrect_number_list = str2list(incorrect_number_record["inflection"])
            incorrect_row_feats = {
                feat: incorrect_number_record[feat]
                for feat in self.mismatched_features
                if feat in incorrect_number_record
            }

            for incorrect_number in incorrect_number_list:
                incorrect_sentence = self.sentence_template.render(
                    number=incorrect_number,
                    noun=noun_features["inflection"],
                    scenario="judge",
                ).strip()
                incorrect_prompt = self.judge_template.render(
                    language=self.language,
                    sentence=incorrect_sentence,
                    additional_instruction=additional_instruction,
                )
                mismatched_feats = self._get_mismatched_features(
                    incorrect_number_record, number_features
                )

                sentence_data = {
                    "id": f"{base_number}_{example_id}_neg",
                    "prompt": incorrect_prompt,
                    "gold": "No",
                    "metadata": {
                        "language": self.language,
                        "lang_code": self.lang_code,
                        "scenario": "judge",
                        "template_name": self.template_name,
                        "sentence": incorrect_sentence,
                        "slot_features": {
                            "number": {
                                "base_word": incorrect_number_record["base_word"],
                                "inflection": incorrect_number,
                                **incorrect_row_feats,
                            },
                            "noun": noun_features,
                        },
                        "correct_inflections": correct_inflections,
                        "mismatched_features": mismatched_feats,
                    },
                }
                examples.append(sentence_data)
                example_id += 1
        return examples, example_id

    def _apply_correct_inflection_logic(self, features_dict: Dict) -> Dict:
        """
        Apply Hebrew-specific noun inflection logic for plural cardinal numbers. If the
        number is between 3-10, its gender should NOT contradict  the gender of the following noun

        Args:
            features_dict (Dict): Feature dictionary.

        Returns:
            Dict: Updated feature dictionary with correct gender and plurality.
        """
        features_dict = features_dict.copy()
        return features_dict
