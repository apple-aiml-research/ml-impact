#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pandas as pd
from typing import List, Dict, Tuple

from src.templates.template_base import TemplateBase, register_template


@register_template
class RusMotionVerbs(TemplateBase):
    """Generates data for motion verbs"""

    def __init__(self):
        self.setup_common(template_name="motion_verbs.j2", lang_code="rus")

        self.features = []
        self.additional_features = {"case": "ACC"}
        self.mismatched_features = ["direction"]

        self.base_places = self.placeholders.get_by_features(
            "places",
        ).base_word.unique()

        self.motion_verbs_df = self.placeholders.get_by_features(
            "verbs_motion", **{"plurality": "SG"}
        )
        self.time = {"unidirectional": "сейчас", "multidirectional": "каждый день"}

    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        example_id = 1

        for direction in self.time:
            direction_motion_df = self.motion_verbs_df[
                self.motion_verbs_df.direction == direction
            ]
            time_adverb = self.time[direction]
            for _, row in direction_motion_df.iterrows():
                base_verb, base_correct_motion_verb, correct_inflection = (
                    row["base_word"],
                    row["direction"],
                    row["inflection"],
                )
                for base_place in self.base_places:
                    place_feature_filter = {
                        **self.additional_features,
                        "base_word": base_place,
                    }

                    correct_inflected_places_row = self.placeholders.get_by_features(
                        "places", **place_feature_filter
                    ).squeeze()
                    correct_inflected_places = self.get_inflected_forms(
                        "places", place_feature_filter
                    )
                    preposition = correct_inflected_places_row["preposition"].strip()
                    if preposition:
                        preposition = preposition + " "

                    for correct_inflected_place in correct_inflected_places:
                        correct_sentence = self.sentence_template.render(
                            place=correct_inflected_place,
                            preposition=preposition,
                            time=time_adverb,
                            scenario="generation",
                        ).strip()
                        additional_instruction = "Make sure that the verb form sounds natural to a native speaker."
                        correct_prompt = self.generation_template.render(
                            language=self.language,
                            pos=base_verb,
                            pos_name="verb",
                            sentence=correct_sentence,
                            additional_instruction=additional_instruction,
                        )

                        sentence_data = {
                            "id": f"{base_correct_motion_verb}_{base_place}_{example_id}_gen",
                            "prompt": correct_prompt,
                            "gold": [correct_inflection],
                            "metadata": {
                                "language": self.language,
                                "lang_code": self.lang_code,
                                "scenario": "generation",
                                "template_name": self.template_name,
                                "sentence": correct_sentence,
                                "slot_features": {
                                    "place": place_feature_filter,
                                    "verb": {
                                        "base_word": base_verb,
                                        "directionality": direction,
                                        "base_motion_verb": base_correct_motion_verb,
                                        "plurality": "SG",
                                        "inflection": correct_inflection,
                                    },
                                },
                            },
                        }

                        examples.append(sentence_data)
                        example_id += 1

        return examples

    def _generate_judgment(self, additional_instruction: str = "") -> List[Dict]:
        examples = []
        example_id = 1

        for _, correct_row in self.motion_verbs_df.iterrows():
            direction = correct_row["direction"]
            for base_place in self.base_places:
                (
                    correct_examples,
                    correct_verb,
                    example_id,
                ) = self._create_correct_judgments(
                    correct_row, base_place, additional_instruction, example_id
                )
                examples.extend(correct_examples)
                incorrect_row = self.motion_verbs_df[
                    (self.motion_verbs_df.base_word == correct_row["base_word"])
                    & (self.motion_verbs_df.direction != direction)
                ].squeeze()
                (
                    incorrect_examples,
                    example_id,
                ) = self._create_incorrect_judgments(
                    correct_verb,
                    incorrect_row,
                    direction,
                    base_place,
                    additional_instruction,
                    example_id,
                )
                examples.extend(incorrect_examples)

        return examples

    def _create_correct_judgments(
        self,
        correct_row: pd.Series,
        base_place: str,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], List[str], int]:
        """
        Generate correct judgment examples for motion verb test.

        Args:
            correct_row (pd.Series): correct df row
            base_place (str): noun place
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], List[str], Dict, int]: Correct examples, correct inflections, feature filter, and updated example ID.
        """

        examples = []
        base_verb, base_correct_motion_verb, correct_verb = (
            correct_row["base_word"],
            correct_row["base_motion_verb"],
            correct_row["inflection"],
        )
        direction = correct_row["direction"]
        time_adverb = self.time[direction]

        place_feature_filter = {
            **self.additional_features,
            "base_word": base_place,
        }
        correct_inflected_places = self.get_inflected_forms(
            "places", place_feature_filter
        )

        correct_inflected_places_row = self.placeholders.get_by_features(
            "places", **place_feature_filter
        ).squeeze()
        correct_inflected_places = self.get_inflected_forms(
            "places", place_feature_filter
        )
        preposition = correct_inflected_places_row["preposition"]
        if preposition:
            preposition = preposition + " "

        for correct_inflected_place in correct_inflected_places:
            correct_sentence = self.sentence_template.render(
                place=correct_inflected_place,
                verb=correct_verb,
                preposition=preposition,
                time=time_adverb,
                scenario="judge",
            ).strip()
            additional_instruction = (
                "Make sure that the verb form sounds natural to a native speaker."
            )
            correct_prompt = self.judge_template.render(
                language=self.language,
                pos=base_verb,
                pos_name="verb",
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{direction}_{base_verb}_{example_id}_pos",
                "prompt": correct_prompt,
                "gold": "Yes",
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "judge",
                    "template_name": self.template_name,
                    "sentence": correct_sentence,
                    "slot_features": {
                        "place": place_feature_filter,
                        "verb": {
                            "base_word": base_verb,
                            "directionality": direction,
                            "base_motion_verb": base_correct_motion_verb,
                            "plurality": "SG",
                            "inflection": correct_verb,
                        },
                    },
                    "correct_inflections": [correct_verb],
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples, [correct_verb], example_id

    def _create_incorrect_judgments(
        self,
        correct_verbs: List[str],
        incorrect_row: pd.Series,
        direction: str,
        base_place: str,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], int]:
        """
        Generate incorrect judgment examples for motion verbs.

        Args:
            correct_verbs (Lisr[str]): correct directional verb list
            incorrect_row (pd.Series): incorrect direction row
            direction (str): motion verb direction
            base_place (str): noun place
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], int]: Incorrect examples and updated example ID.
        """
        examples = []

        base_verb, base_incorrect_motion_verb, incorrect_verb = (
            incorrect_row["base_word"],
            incorrect_row["base_motion_verb"],
            incorrect_row["inflection"],
        )
        time_adverb = self.time[direction]

        place_feature_filter = {
            **self.additional_features,
            "base_word": base_place,
        }
        correct_inflected_places_row = self.placeholders.get_by_features(
            "places", **place_feature_filter
        ).squeeze()
        correct_inflected_places = self.get_inflected_forms(
            "places", place_feature_filter
        )
        preposition = correct_inflected_places_row["preposition"]
        if preposition:
            preposition = preposition + " "

        for correct_inflected_place in correct_inflected_places:
            incorrect_sentence = self.sentence_template.render(
                place=correct_inflected_place,
                verb=incorrect_verb,
                preposition=preposition,
                time=time_adverb,
                scenario="judge",
            ).strip()
            additional_instruction = (
                "Make sure that the verb form sounds natural to a native speaker."
            )
            incorrect_prompt = self.judge_template.render(
                language=self.language,
                pos=base_verb,
                pos_name="verb",
                sentence=incorrect_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{direction}_{base_verb}_{example_id}_neg",
                "prompt": incorrect_prompt,
                "gold": "No",
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "judge",
                    "template_name": self.template_name,
                    "sentence": incorrect_sentence,
                    "slot_features": {
                        "place": place_feature_filter,
                        "verb": {
                            "base_word": base_verb,
                            "directionality": incorrect_row["direction"],
                            "base_motion_verb": base_incorrect_motion_verb,
                            "plurality": "SG",
                            "inflection": incorrect_verb,
                        },
                    },
                    "correct_inflections": correct_verbs,
                    "mismatched_features": self.mismatched_features,
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples, example_id
