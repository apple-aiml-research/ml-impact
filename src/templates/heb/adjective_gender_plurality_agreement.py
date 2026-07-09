#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import random
import pandas as pd
from typing import List, Dict, Tuple

from src.utils.text import str2list
from src.templates.template_base import TemplateBase, register_template


@register_template
class HebAdjectiveGenderPluralityAgreement(TemplateBase):
    """Generates data for noun-adjective agreement in Hebrew sentences based on gender, and plurality."""

    def __init__(self):
        self.setup_common(
            template_name="adjective_gender_plurality_agreement.j2", lang_code="heb"
        )

        self.features = ["gender", "plurality", "animacy"]
        self.additional_features = {}
        self.mismatched_features = ["gender", "plurality"]

        # Cache noun data
        self.noun_cache = {
            "occupations": {
                "MASC": self.placeholders.get_by_features(
                    "occupations", **{"gender": "MASC", "animacy": "HUM"}
                ),
                "FEM": self.placeholders.get_by_features(
                    "occupations", **{"gender": "FEM", "animacy": "HUM"}
                ),
            },
            "objects": {
                "MASC": self.placeholders.get_by_features(
                    "objects", **{"gender": "MASC", "animacy": "NHUM"}
                ),
                "FEM": self.placeholders.get_by_features(
                    "objects", **{"gender": "FEM", "animacy": "NHUM"}
                ),
            },
        }

    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        examples = {}

        feature_combinations, _ = self.placeholders.get_feature_combinations(
            self.features, "adjectives"
        )
        example_id = 1

        for feature_combination in feature_combinations:
            noun_features = dict(zip(self.features, feature_combination))
            animacy = noun_features["animacy"]
            noun_placeholder = "occupations" if animacy == "HUM" else "objects"

            noun_data = self.noun_cache[noun_placeholder][noun_features["gender"]]
            base_nouns = noun_data.base_word.unique()
            base_adjectives = self.placeholders.get_by_features(
                "adjectives", animacy=animacy, **self.additional_features
            ).base_word.unique()

            for base_noun in base_nouns:
                noun_feature_filter = {**noun_features, "base_word": base_noun}
                inflected_nouns = self.get_inflected_forms(
                    noun_placeholder, noun_feature_filter
                )

                for inflected_noun in inflected_nouns:
                    for base_adjective in base_adjectives:
                        adj_feature_filter = {
                            **noun_features,
                            "base_word": base_adjective,
                            **self.additional_features,
                        }
                        adj_feature_filter = self._apply_correct_inflection_logic(
                            adj_feature_filter
                        )  # apply adjective agreement logic
                        inflected_adjectives = self.get_inflected_forms(
                            "adjectives", adj_feature_filter
                        )

                        correct_sentence = self.sentence_template.render(
                            noun=inflected_noun, scenario="generation"
                        ).strip()
                        correct_prompt = self.generation_template.render(
                            language=self.language,
                            pos=base_adjective,
                            pos_name="adjective",
                            sentence=correct_sentence,
                            additional_instruction=additional_instruction,
                        )

                        sentence_data = {
                            "id": f"{base_adjective}_{example_id}_gen",
                            "prompt": correct_prompt,
                            "gold": inflected_adjectives,
                            "metadata": {
                                "language": self.language,
                                "lang_code": self.lang_code,
                                "scenario": "generation",
                                "template_name": self.template_name,
                                "sentence": correct_sentence,
                                "slot_features": {
                                    "noun": {
                                        **noun_feature_filter,
                                        "inflection": inflected_noun,
                                    },
                                    "adjective": {
                                        **adj_feature_filter,
                                        "inflection": inflected_adjectives,
                                    },
                                },
                            },
                        }
                        key = correct_prompt
                        examples[key] = examples.get(key, []) + [sentence_data]
                        example_id += 1
        post_processed_examples = self.postprocess_examples_generation(examples)
        return post_processed_examples

    def _generate_judgment(self, additional_instruction: str = "") -> List[Dict]:
        examples = {}

        feature_combinations, _ = self.placeholders.get_feature_combinations(
            self.features, "adjectives"
        )
        example_id = 1

        for feature_combination in feature_combinations:
            noun_features = dict(zip(self.features, feature_combination))
            animacy = noun_features["animacy"]
            noun_placeholder = "occupations" if animacy == "HUM" else "objects"

            noun_data = self.noun_cache[noun_placeholder][noun_features["gender"]]
            base_nouns = noun_data.base_word.unique()
            base_adjectives = self.placeholders.get_by_features(
                "adjectives", animacy=animacy, **self.additional_features
            ).base_word.unique()

            for base_noun in base_nouns:
                noun_feature_filter = {**noun_features, "base_word": base_noun}
                inflected_nouns = self.get_inflected_forms(
                    noun_placeholder, noun_feature_filter
                )

                for inflected_noun in inflected_nouns:
                    noun_features_full = {
                        **noun_feature_filter,
                        "inflection": inflected_noun,
                    }

                    for base_adjective in base_adjectives:
                        (
                            correct_examples,
                            correct_inflection,
                            adj_feature_filter,
                            example_id,
                        ) = self._create_correct_judgments(
                            noun_features_full,
                            base_adjective,
                            additional_instruction,
                            example_id,
                        )

                        for correct_example in correct_examples:
                            key = correct_example["prompt"]
                            if key not in examples:
                                examples[key] = {}
                            examples[key]["Yes"] = examples[key].get("Yes", []) + [
                                correct_example
                            ]
                        # we need to remove the features of adjectives
                        # that might have other inflections
                        exclude_features = {
                            "gender": [adj_feature_filter["gender"]],
                            "plurality": [adj_feature_filter["plurality"]],
                        }
                        (
                            incorrect_examples,
                            example_id,
                        ) = self._create_incorrect_judgments(
                            noun_features_full,
                            base_adjective,
                            adj_feature_filter,
                            correct_inflection,
                            additional_instruction,
                            example_id,
                            exclude_features,
                        )

                        for incorrect_example in incorrect_examples:
                            key = incorrect_example["prompt"]
                            if key not in examples:
                                examples[key] = {}
                            examples[key]["No"] = examples[key].get("No", []) + [
                                incorrect_example
                            ]
        postprocessed_examples = self.postprocess_examples_judge(examples)
        return postprocessed_examples

    def _create_correct_judgments(
        self,
        noun_features: Dict,
        base_adjective: str,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], List[str], Dict, int]:
        """
        Generate correct judgment examples for adjective agreement.

        Args:
            noun_features (Dict): Noun features (base_word, inflected_form, gender, plurality, animacy).
            base_adjective (str): Root form of the adjective.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], List[str], Dict, int]: Correct examples, correct inflections, feature filter, and updated example ID.
        """

        examples = []
        adj_feature_filter = {
            k: v for (k, v) in noun_features.items() if k in self.features
        }
        adj_feature_filter = {
            **adj_feature_filter,
            **self.additional_features,
            "base_word": base_adjective,
        }
        adj_feature_filter = self._apply_correct_inflection_logic(adj_feature_filter)
        correct_inflections = self.get_inflected_forms("adjectives", adj_feature_filter)

        for correct_inflection in correct_inflections:
            correct_sentence = self.sentence_template.render(
                noun=noun_features["inflection"],
                adjective=correct_inflection,
                scenario="judge",
            ).strip()
            correct_prompt = self.judge_template.render(
                language=self.language,
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{base_adjective}_{example_id}_pos",
                "prompt": correct_prompt,
                "gold": "Yes",
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "judge",
                    "template_name": self.template_name,
                    "sentence": correct_sentence,
                    "slot_features": {
                        "noun": noun_features,
                        "adjective": {
                            **adj_feature_filter,
                            "inflection": correct_inflection,
                        },
                    },
                    "correct_inflections": correct_inflections,
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples, correct_inflections, adj_feature_filter, example_id

    def _create_incorrect_judgments(
        self,
        noun_features: Dict,
        base_adjective: str,
        adjective_features: Dict,
        correct_inflections: List[str],
        additional_instruction: str,
        example_id: int,
        exclude_features: Dict = None,
    ) -> Tuple[List[Dict], int]:
        """
        Generate incorrect judgment examples for adjective agreement, excluding specified feature values or lists.

        Args:
            noun_features (Dict): Noun features (base_word, inflected_form, gender, plurality, animacy).
            base_adjective (str): Root form of the adjective.
            adjective_features (Dict): Adjective features (gender, plurality, animacy).
            correct_inflections (List[str]): List of correct adjective inflections.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.
            exclude_features (Dict, optional): Features to exclude (e.g., {'gender': ['MASC']}).

        Returns:
            Tuple[List[Dict], int]: Incorrect examples and updated example ID.
        """

        examples = []
        # generate incorrect examples
        adj_df = self.get_inflected_forms(
            "adjectives",
            {
                "base_word": base_adjective,
                "animacy": noun_features.get("animacy"),
                **self.additional_features,
            },
            return_df=True,
        )

        correct_inflection_set = set(correct_inflections)
        # We now have the correct inflection set and need to get the incorrect one
        # We can do this by filtering out the inflections that are different
        # But before, we might have scenarios where a different inflection is actually correct

        exclude_condition = pd.Series([True] * len(adj_df), index=adj_df.index)

        conditions = []
        if exclude_features:
            for feat, value in exclude_features.items():
                if feat not in adj_df.columns:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    conditions.append(adj_df[feat].isin(value))
                else:
                    conditions.append(adj_df[feat] == value)

        if conditions:
            exclude_condition = ~pd.concat(conditions, axis=1).all(axis=1)

        incorrect_adjective_data = adj_df[
            exclude_condition
            & ~adj_df["inflection"].apply(
                lambda x: any(v in correct_inflection_set for v in str2list(x))
            )
        ]

        incorrect_adjective_data = incorrect_adjective_data.drop_duplicates(
            "inflection"
        )  # rm incorrect data having same inflections

        if incorrect_adjective_data.empty:
            raise ValueError(
                f"There are no left inflected forms for {base_adjective} with {correct_inflection_set}."
            )

        for _, incorrect_adjective_record in incorrect_adjective_data.iterrows():
            incorrect_inflections = str2list(incorrect_adjective_record["inflection"])
            incorrect_adjective_features = {
                feat: incorrect_adjective_record[feat]
                for feat in self.mismatched_features
                if feat in incorrect_adjective_record
            }

            for incorrect_inflection in incorrect_inflections:
                incorrect_sentence = self.sentence_template.render(
                    noun=noun_features["inflection"],
                    adjective=incorrect_inflection,
                    scenario="judge",
                ).strip()
                incorrect_prompt = self.judge_template.render(
                    language=self.language,
                    sentence=incorrect_sentence,
                    additional_instruction=additional_instruction,
                )
                mismatched_features = self._get_mismatched_features(
                    incorrect_adjective_record, adjective_features
                )

                sentence_data = {
                    "id": f"{base_adjective}_{example_id}_neg",
                    "prompt": incorrect_prompt,
                    "gold": "No",
                    "metadata": {
                        "language": self.language,
                        "lang_code": self.lang_code,
                        "scenario": "judge",
                        "template_name": self.template_name,
                        "sentence": incorrect_sentence,
                        "slot_features": {
                            "noun": noun_features,
                            "adjective": {
                                "base_word": incorrect_adjective_record["base_word"],
                                "inflection": incorrect_inflection,
                                **incorrect_adjective_features,
                                **self.additional_features,
                            },
                        },
                        "correct_inflections": correct_inflections,
                        "mismatched_features": mismatched_features,
                    },
                }
                examples.append(sentence_data)
                example_id += 1

        return examples, example_id

    def postprocess_examples_generation(self, examples: Dict) -> List[Dict]:
        # examples where noun inflection is the same
        # in that case group the inflections for adjectives
        post_processed_examples = []

        for key in examples:
            datapoints = examples[key]
            if len(datapoints) == 1:
                datapoint2add = datapoints[0]
            else:
                # group all inflected aspects into gold
                # choose random example and append
                random.seed(42)
                datapoint2add = random.choice(datapoints)
                combined_gold = [gold for x in datapoints for gold in x["gold"]]
                datapoint2add["gold"] = list(set(combined_gold))

            post_processed_examples.append(datapoint2add)
        # Return the grouped examples
        return post_processed_examples

    def postprocess_examples_judge(self, examples: Dict) -> List[Dict]:
        # sometimes we have examples that are grouped as "Yes" and "No".
        # in that case, take one random examples feom yes "Yes"
        post_processed_examples = []

        for key in examples:
            answer_dict = examples[key]
            if len(answer_dict) == 1:
                for _, vals in answer_dict.items():
                    random.seed(42)
                    datapoint2add = random.choice(vals)

            else:
                # take yes answer
                random.seed(42)
                datapoint2add = random.choice(answer_dict["Yes"])
            post_processed_examples.append(datapoint2add)
        return post_processed_examples

    def _apply_correct_inflection_logic(self, features_dict: Dict) -> Dict:
        """
        Apply Hebrew-specific adjective inflection logic for inanimate plural nouns. If the noun is plural and inanimate,
        the adjective should be singular feminine.

        Args:
            features_dict (Dict): Feature dictionary for adjective or noun.

        Returns:
            Dict: Updated feature dictionary with correct gender and plurality for adjectives.
        """
        features_dict = features_dict.copy()
        features_dict.pop("animacy", None)
        return features_dict
