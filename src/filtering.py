from typing import List, Optional, Set

import pandas as pd


def _contains_filter_value(values: List[str], target: str) -> bool:
    """
    Case-insensitive exact membership test for metadata include filters.
    """
    if not isinstance(values, list):
        return False
    if not isinstance(target, str):
        return False

    target = target.strip().lower()
    values = [str(v).strip().lower() for v in values]
    return target in values


def _contains_any_excluded_value(values: List[str], excluded_values: List[str]) -> bool:
    """
    Case-insensitive overlap test for metadata exclude filters.
    """
    if not isinstance(values, list):
        return False

    value_set = {str(v).strip().lower() for v in values}
    excluded_set = {str(v).strip().lower() for v in excluded_values}
    return len(value_set.intersection(excluded_set)) > 0


def build_filter_mask(
    results_df: pd.DataFrame,
    recipe_ingredient_sets: List[Set[str]],
    normalize_ingredient_text_func,
    include_cuisine: Optional[str] = None,
    exclude_cuisines: Optional[List[str]] = None,
    meal_type: Optional[str] = None,
    dish_type: Optional[str] = None,
    diet_label: Optional[str] = None,
    min_calories: Optional[float] = None,
    max_calories: Optional[float] = None,
    exclude_ingredients: Optional[List[str]] = None,
) -> pd.Series:
    """
    Build a boolean mask based on optional metadata / ingredient exclusion filters.
    If no filters are provided, all rows remain included.
    """
    mask = pd.Series(True, index=results_df.index)

    if include_cuisine is not None and "cuisine_type" in results_df.columns:
        mask &= results_df["cuisine_type"].apply(
            lambda x: _contains_filter_value(x, include_cuisine)
        )

    if exclude_cuisines and "cuisine_type" in results_df.columns:
        normalized_excluded_cuisines = [
            c.strip().lower() for c in exclude_cuisines if str(c).strip()
        ]
        mask &= ~results_df["cuisine_type"].apply(
            lambda x: _contains_any_excluded_value(x, normalized_excluded_cuisines)
        )

    if meal_type is not None and "meal_type" in results_df.columns:
        mask &= results_df["meal_type"].apply(
            lambda x: _contains_filter_value(x, meal_type)
        )

    if dish_type is not None and "dish_type" in results_df.columns:
        mask &= results_df["dish_type"].apply(
            lambda x: _contains_filter_value(x, dish_type)
        )

    if diet_label is not None and "diet_labels" in results_df.columns:
        mask &= results_df["diet_labels"].apply(
            lambda x: _contains_filter_value(x, diet_label)
        )

    if min_calories is not None or max_calories is not None:
        calories = pd.to_numeric(results_df["calories"], errors="coerce")

        if min_calories is not None:
            mask &= calories >= min_calories

        if max_calories is not None:
            mask &= calories <= max_calories

    if exclude_ingredients:
        normalized_excluded_ingredients = []
        for ing in exclude_ingredients:
            norm = normalize_ingredient_text_func(ing)
            if norm:
                normalized_excluded_ingredients.append(norm)

        excluded_set = set(normalized_excluded_ingredients)

        ingredient_mask = []
        for ingredient_set in recipe_ingredient_sets:
            ingredient_mask.append(len(ingredient_set.intersection(excluded_set)) == 0)

        mask &= pd.Series(ingredient_mask, index=results_df.index)

    return mask


def apply_match_constraints(
    results_df: pd.DataFrame,
    query_size: int,
    require_all_query_matches: bool = False,
    min_match_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply post-retrieval constraints related to how many query ingredients matched.
    """
    filtered = results_df.copy()

    if require_all_query_matches:
        filtered = filtered[filtered["matched_count"] == query_size].copy()

    if min_match_count is not None:
        filtered = filtered[filtered["matched_count"] >= min_match_count].copy()

    return filtered