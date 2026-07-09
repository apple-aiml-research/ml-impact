#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import random
import pandas as pd
from typing import List, Dict, Tuple

from src.utils.text import generate_noun_variations, str2list
from src.templates.template_base import TemplateBase, register_template


@register_template
class EngVerbPluralityAgreementImperative(TemplateBase):
    """Generate data for noun-verb (imperative) agreement in English sentences based on plurality."""

    def __init__(self):
        self.setup_common(
            template_name="verb_plurality_agreement_imperative.j2",
            lang_code="eng",
        )

        self.features = [
            "plurality"
        ]  # these are features we permute to generate test cases
        self.additional_features = {
            "mood": "IMP"
        }  # these are features we do not permute but need in order to filter verbs
        self.mismatched_features = self.features

        # Cache imperative verbs
        self.base_imperative_verbs = self.placeholders.get_by_features(
            "verbs", **self.additional_features
        ).base_word.unique()

    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        examples = {}

        feature_combinations, _ = self.placeholders.get_feature_combinations(
            self.features, "verbs"
        )
        example_id = 1

        for feature_combination in feature_combinations:
            verb_features = dict(zip(self.features, feature_combination))
            verb_features.update(self.additional_features)
            noun_plurality = verb_features["plurality"]  # verb agrees with noun

            person_names = self.placeholders.get_by_features("names").name.tolist()
            name_groups = generate_noun_variations(person_names, noun_plurality)

            for name_group in name_groups:
                name_text = (
                    ", ".join(name_group[:-1]) + "," + self.conjunction + name_group[-1]
                    if noun_plurality != "SG"
                    else name_group[0]
                ).strip()
                noun_features = {
                    "name": name_text,
                    "plurality": noun_plurality,
                }

                for root_verb in self.base_imperative_verbs:
                    verb_feature_filter = {**verb_features, "base_word": root_verb}
                    inflected_forms = self.get_inflected_forms(
                        "verbs", verb_feature_filter
                    )

                    correct_sentence = self.sentence_template.render(
                        name=name_text, scenario="generation"
                    ).strip()
                    additional_instruction = (
                        "The inflected verb should be in the imperative form."
                    )
                    correct_prompt = self.generation_template.render(
                        language=self.language,
                        pos=root_verb,
                        pos_name="verb",
                        sentence=correct_sentence,
                        additional_instruction=additional_instruction,
                    )

                    sentence_data = {
                        "id": f"{root_verb}_{example_id}_gen",
                        "prompt": correct_prompt,
                        "gold": inflected_forms,
                        "metadata": {
                            "language": self.language,
                            "lang_code": self.lang_code,
                            "scenario": "generation",
                            "template_name": self.template_name,
                            "sentence": correct_sentence,
                            "slot_features": {
                                "noun": noun_features,
                                "verb": {
                                    **verb_feature_filter,
                                    "inflection": inflected_forms,
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
            self.features, "verbs"
        )

        example_id = 1

        for feature_combination in feature_combinations:
            verb_features = dict(zip(self.features, feature_combination))
            verb_features.update(self.additional_features)
            noun_plurality = verb_features["plurality"]  # verb agrees with noun

            person_names = self.placeholders.get_by_features("names").name.tolist()
            name_groups = generate_noun_variations(person_names, noun_plurality)

            for name_group in name_groups:
                name_text = (
                    ", ".join(name_group[:-1]) + "," + self.conjunction + name_group[-1]
                    if noun_plurality != "SG"
                    else name_group[0]
                ).strip()
                noun_features = {
                    "name": name_text,
                    "plurality": noun_plurality,
                }

                for root_verb in self.base_imperative_verbs:
                    # generate correct example
                    (
                        correct_examples,
                        correct_inflections,
                        example_id,
                    ) = self._create_correct_judgments(
                        noun_features,
                        root_verb,
                        verb_features,
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
                    exclude_features = {
                        "plurality": [noun_plurality],
                    }

                    incorrect_examples, example_id = self._create_incorrect_judgments(
                        noun_features,
                        root_verb,
                        verb_features,
                        correct_inflections,
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
        root_verb: str,
        verb_features: Dict,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], List[str], int]:
        """
        Generate correct judgment examples for verb agreement.

        Args:
            noun_features (Dict): Noun features (name, gender, plurality).
            root_verb (str): Root form of the verb.
            verb_features (Dict): Verb features (gender, plurality).
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], List[str], int]: Correct examples, correct inflections, and updated example ID.
        """

        examples = []
        verb_feature_filter = {
            **verb_features,
            "base_word": root_verb,
        }
        correct_inflections = self.get_inflected_forms("verbs", verb_feature_filter)

        for correct_inflection in correct_inflections:
            correct_sentence = self.sentence_template.render(
                name=noun_features["name"], verb=correct_inflection, scenario="judge"
            ).strip()
            correct_prompt = self.judge_template.render(
                language=self.language,
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{root_verb}_{example_id}_pos",
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
                        "verb": {
                            **verb_feature_filter,
                            "inflection": correct_inflection,
                        },
                    },
                    "correct_inflections": correct_inflections,
                },
            }
            examples.append(sentence_data)
            example_id += 1
        return examples, correct_inflections, example_id

    def _create_incorrect_judgments(
        self,
        noun_features: Dict,
        root_verb: str,
        verb_features: Dict,
        correct_inflections: List[str],
        additional_instruction: str,
        example_id: int,
        exclude_features: Dict = None,
    ) -> Tuple[List[Dict], int]:
        """
        Generate incorrect judgment examples for verb agreement, excluding specified feature values or lists.

        Args:
            noun_features (Dict): Noun features (name, gender, plurality).
            root_verb (str): Root form of the verb.
            verb_features (Dict): Verb features (gender, plurality).
            correct_inflections (List[str]): List of correct verb inflections.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.
            exclude_features (Dict, optional): Features to exclude (single values or lists, e.g., {'gender': ['MASC']}).

        Returns:
            Tuple[List[Dict], int]: Incorrect examples and updated example ID.
        """

        examples = []
        verb_df = self.get_inflected_forms(
            "verbs",
            {"base_word": root_verb, **self.additional_features},
            return_df=True,
        )
        correct_inflection_set = set(correct_inflections)
        # We now have the correct inflection set and need to get the incorrect one
        # We can do this by filtering out the inflections that are different
        # But before, we might have scenarios where a different inflection is actually correct
        exclude_condition = pd.Series([True] * len(verb_df), index=verb_df.index)

        conditions = []
        if exclude_features:
            for feat, value in exclude_features.items():
                if feat not in verb_df.columns:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    conditions.append(verb_df[feat].isin(value))
                else:
                    conditions.append(verb_df[feat] == value)

        if conditions:
            exclude_condition = ~pd.concat(conditions, axis=1).all(axis=1)

        incorrect_verb_data = verb_df[
            exclude_condition
            & ~verb_df["inflection"].apply(
                lambda x: any(v in correct_inflection_set for v in str2list(x))
            )
        ]

        if incorrect_verb_data.empty:
            return examples, example_id

        incorrect_verb_data = incorrect_verb_data.drop_duplicates(
            "inflection"
        )  # rm incorrect data having same inflections

        for _, incorrect_verb_row in incorrect_verb_data.iterrows():
            incorrect_inflections = str2list(incorrect_verb_row["inflection"])
            incorrect_verb_features = {
                feat: incorrect_verb_row[feat]
                for feat in self.features
                if feat in incorrect_verb_row
            }

            for incorrect_inflection in incorrect_inflections:
                incorrect_sentence = self.sentence_template.render(
                    name=noun_features["name"],
                    verb=incorrect_inflection,
                    scenario="judge",
                ).strip()
                incorrect_prompt = self.judge_template.render(
                    language=self.language,
                    sentence=incorrect_sentence,
                    additional_instruction=additional_instruction,
                )
                mismatched_features = self._get_mismatched_features(
                    incorrect_verb_row, verb_features
                )

                sentence_data = {
                    "id": f"{root_verb}_{example_id}_neg",
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
                            "verb": {
                                "base_word": incorrect_verb_row["base_word"],
                                "inflection": incorrect_inflection,
                                **incorrect_verb_features,
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
