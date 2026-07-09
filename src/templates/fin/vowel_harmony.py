#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import List, Dict, Tuple

from src.templates.template_base import TemplateBase, register_template


@register_template
class FinVowelHarmony(TemplateBase):
    """Generates data for vowel harmony"""

    def __init__(self):
        self.setup_common(template_name="vowel_harmony.j2", lang_code="fin")

        self.features = [""]
        self.additional_features = {"case": "elative"}
        self.mismatched_features = self.features

    def _generate_generation(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        example_id = 1

        base_places = self.placeholders.get_by_features(
            "places", **self.additional_features
        ).base_word.unique()
        additional_instruction = 'The verb "pitää" here means "to like", so use the correct noun form that follows this meaning.'
        for base_place in base_places:
            place_feature_filter = {
                **self.additional_features,
                "base_word": base_place,
            }

            correct_inflected_place = self.get_inflected_forms(
                "places", place_feature_filter
            )

            correct_sentence = self.sentence_template.render(
                scenario="generation"
            ).strip()
            correct_prompt = self.generation_template.render(
                language=self.language,
                pos=base_place,
                pos_name="noun",
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{base_place}_{example_id}_gen",
                "prompt": correct_prompt,
                "gold": correct_inflected_place,
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "generation",
                    "template_name": self.template_name,
                    "sentence": correct_sentence,
                    "slot_features": {
                        "place": place_feature_filter,
                    },
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples

    def _generate_judgment(self, additional_instruction: str = "") -> List[Dict]:
        examples = []

        example_id = 1

        base_places = self.placeholders.get_by_features(
            "places", **self.additional_features
        ).base_word.unique()

        additional_instruction = 'The verb "pitää" here means "to like", so use the correct noun form that follows this meaning.'

        for base_place in base_places:
            (
                correct_examples,
                correct_inflections,
                example_id,
            ) = self._create_correct_judgments(
                base_place,
                additional_instruction,
                example_id,
            )
            examples.extend(correct_examples)

            (
                incorrect_examples,
                example_id,
            ) = self._create_incorrect_judgments(
                base_place,
                correct_inflections,
                additional_instruction,
                example_id,
            )
            examples.extend(incorrect_examples)

        return examples

    def _create_correct_judgments(
        self,
        base_place: str,
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], List[str], int]:
        """
        Generate correct judgment examples for vowel harmony.

        Args:
            base_place (str): noun place
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], List[str], int]: Correct examples, correct inflections, and updated example ID.
        """

        examples = []

        place_feature_filter = {
            **self.additional_features,
            "base_word": base_place,
        }

        correct_inflections = self.get_inflected_forms("places", place_feature_filter)

        for correct_inflection in correct_inflections:
            correct_sentence = self.sentence_template.render(
                place=correct_inflection,
                scenario="judge",
            ).strip()
            correct_prompt = self.judge_template.render(
                language=self.language,
                sentence=correct_sentence,
                additional_instruction=additional_instruction,
            )

            sentence_data = {
                "id": f"{base_place}_{example_id}_pos",
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
                    },
                    "correct_inflections": correct_inflections,
                },
            }

            examples.append(sentence_data)
            example_id += 1

        return examples, correct_inflections, example_id

    def _create_incorrect_judgments(
        self,
        base_place: str,
        correct_inflections: List[str],
        additional_instruction: str,
        example_id: int,
    ) -> Tuple[List[Dict], int]:
        """
        Generate incorrect judgment examples for vowel harmony, excluding specified feature values or lists.

        Args:
            base_place (str): masculine form of the number.
            correct_inflections (List[str]): List of correct number inflections.
            additional_instruction (str): Additional prompt instructions.
            example_id (int): Current example ID counter.

        Returns:
            Tuple[List[Dict], int]: Incorrect examples and updated example ID.
        """

        examples = []

        incorrect_places = [self.flip_vowel(w) for w in correct_inflections]

        for incorrect_place in incorrect_places:
            incorrect_sentence = self.sentence_template.render(
                place=incorrect_place,
                scenario="judge",
            ).strip()
            incorrect_prompt = self.judge_template.render(
                language=self.language,
                sentence=incorrect_sentence,
                additional_instruction=additional_instruction,
            )
            mismatched_feats = ["vowel harmony"]

            sentence_data = {
                "id": f"{base_place}_{example_id}_neg",
                "prompt": incorrect_prompt,
                "gold": "No",
                "metadata": {
                    "language": self.language,
                    "lang_code": self.lang_code,
                    "scenario": "judge",
                    "template_name": self.template_name,
                    "sentence": incorrect_sentence,
                    "slot_features": {
                        "place": {"base_word": base_place, **self.additional_features},
                    },
                    "correct_inflections": correct_inflections,
                    "mismatched_features": mismatched_feats,
                },
            }
            examples.append(sentence_data)
            example_id += 1
        return examples, example_id

    def flip_vowel(self, w: str):
        flipped_vowel = "a" if w[-1] == "ä" else "ä"
        flipped_word = w[:-1] + flipped_vowel
        assert w != flipped_word
        return flipped_word
