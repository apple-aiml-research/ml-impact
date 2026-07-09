#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import re
import random
import itertools
from ast import literal_eval


def remove_diacritics(text):
    # Arabic diacritics Unicode range
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
    return re.sub(arabic_diacritics, "", text)


def remove_nikud(text):
    # Hebrew vowel marks (nikud) Unicode range and individual points
    nikud_ranges = [
        (0x0591, 0x05BD),  # Main Hebrew diacritics block
        (0x05BF, 0x05BF),  # Hebrew point Rafe
        (0x05C1, 0x05C2),  # Shin and Sin dots
        (0x05C4, 0x05C5),  # Markers
        (0x05C7, 0x05C7),  # Qamats Qatan
    ]

    def is_nikud(char):
        code = ord(char)
        for start, end in nikud_ranges:
            if start <= code <= end:
                return True
        return False

    return "".join(c for c in text if not is_nikud(c))


def generate_noun_variations(singular_nouns, number, max_combinations=100):
    if number == "SG":
        combos = [(noun,) for noun in singular_nouns]
    elif number == "DU":
        combos = list(itertools.combinations(singular_nouns, 2))
    elif number == "PL":
        combos = list(itertools.combinations(singular_nouns, 3))

    if len(combos) > max_combinations:
        random.seed(42)
        return random.sample(combos, max_combinations)
    return combos


def str2list(s):
    return list(set(literal_eval(s)))


def post_process_response(
    raw_response: str, lang: str, split_str: str = "Final Answer:"
) -> str:
    if not raw_response:
        raw_response = ""
    raw_response = raw_response.split(split_str)[-1].replace("*", "").strip()
    if raw_response and raw_response[-1] == ".":
        raw_response = raw_response[:-1]
    raw_response = raw_response.replace("!", "")
    if lang == "ara":
        raw_response = remove_diacritics(raw_response)
    elif lang == "heb":
        raw_response = remove_nikud(raw_response)
    return raw_response.strip()
