#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pandas as pd
from itertools import product

from .text import remove_diacritics


def filter_inflection(df, filters):
    """Filter inflection rows by features and return the first match, diacritics removed."""
    mask = pd.Series([True] * len(df), index=df.index)
    for f in filters:
        mask &= df.Features.str.contains(f";{f}")
    filtered_df = df[mask]
    if not filtered_df.empty:
        return [remove_diacritics(inf) for inf in filtered_df.Inflection.tolist()]
    else:
        return []


def process_pos_data(base_words, df, categories, output_columns, static_labels=None):
    static_labels = static_labels or []
    output_list = []
    for base_word in base_words:
        base_word_clean = remove_diacritics(base_word)
        temp_df = df[df.Base.apply(lambda x: remove_diacritics(x)) == base_word_clean]

        for combo in product(*categories):
            filters = list(combo)
            inflection = filter_inflection(temp_df, filters)
            row = [base_word_clean] + static_labels + filters + [inflection]
            output_list.append(row)
    return pd.DataFrame(output_list, columns=output_columns)
