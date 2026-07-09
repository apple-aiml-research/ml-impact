#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import pandas as pd
from itertools import product
from typing import List
from src.utils.text import str2list


class PlaceholderLoader:
    def __init__(self, lang_code, placeholder_dir="data/placeholders"):
        self.lang_code = lang_code
        self.placeholder_dir = os.path.join(placeholder_dir, lang_code)
        self._cache = {}

    def load_csv(self, category):
        if category not in self._cache:
            file_path = os.path.join(self.placeholder_dir, f"{category}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Placeholder file '{file_path}' not found.")
            self._cache[category] = pd.read_csv(file_path)
        return self._cache[category]

    def get_by_features(self, category: str | List[str], **features):
        if isinstance(category, str):
            df = self.load_csv(category)
        elif isinstance(category, List):
            dfs = [self.load_csv(c) for c in category]
            df = pd.concat(dfs)
        else:
            raise ValueError("Wrong Input.")
        for col, val in features.items():
            df = df[df[col] == val]
        return df

    def get_random(self, category, n=1):
        df = self.load_csv(category)
        return df.sample(n=n)

    def get_feature_combinations(self, feature_names, pos):
        feature_values = {}
        for feat in feature_names:
            df = self.get_by_features(pos)
            if feat not in df.columns:
                feature_values[feat] = []
            else:
                feature_values[feat] = df[feat].dropna().unique()

        return list(product(*feature_values.values())), feature_values

    def combine_categories(self, categories):
        joined_categories = pd.concat(
            [self.get_by_features(category) for category in categories]
        )
        return joined_categories

    def get_inflected_forms(
        self, category, filter_criteria, return_df=False, ok_empty=False
    ):
        df = self.get_by_features(category, **filter_criteria)
        if df.empty or df["inflection"].isna().all():
            if not ok_empty:
                raise ValueError(
                    f"No valid inflected forms for {category} with features: {filter_criteria}"
                )
            else:
                return []

        return df if return_df else str2list(df.iloc[0]["inflection"])
