from typing import Literal

import pandas as pd


SortOption = Literal[
    "best_match",
    "highest_similarity",
    "fewest_missing",
    "lowest_calories",
]


def compute_final_score(
    row: pd.Series,
    similarity_weight: float = 1.0,
    matched_count_weight: float = 0.08,
    recipe_coverage_weight: float = 0.35,
    missing_count_penalty: float = 0.10,
    pantry_bonus: float = 0.25,
) -> float:
    """
    Compute a final recommendation score from retrieval outputs.

    This is intentionally heuristic / product-facing logic:
    - reward cosine similarity
    - reward more matched ingredients
    - reward higher recipe coverage
    - penalize missing ingredients
    - bonus if the recipe can be made with current pantry only
    """
    similarity = float(row.get("similarity", 0.0))
    matched_count = float(row.get("matched_count", 0.0))
    recipe_coverage = float(row.get("recipe_coverage", 0.0))
    missing_count = float(row.get("missing_count", 0.0))

    score = 0.0
    score += similarity_weight * similarity
    score += matched_count_weight * matched_count
    score += recipe_coverage_weight * recipe_coverage
    score -= missing_count_penalty * missing_count

    if missing_count == 0:
        score += pantry_bonus

    return score


def rerank_results(
    results_df: pd.DataFrame,
    sort_by: SortOption = "best_match",
) -> pd.DataFrame:
    """
    Rerank retrieved recipes for the final user-facing display.

    sort_by options:
    - best_match: final weighted score
    - highest_similarity: raw cosine similarity first
    - fewest_missing: recipes needing fewer extra ingredients first
    - lowest_calories: lower-calorie recipes first
    """
    if results_df.empty:
        return results_df.copy()

    df = results_df.copy()

    if "calories" in df.columns:
        df["calories_numeric"] = pd.to_numeric(df["calories"], errors="coerce")
    else:
        df["calories_numeric"] = pd.NA

    df["final_score"] = df.apply(compute_final_score, axis=1)

    if sort_by == "highest_similarity":
        df = df.sort_values(
            by=[
                "similarity",
                "matched_count",
                "recipe_coverage",
                "missing_count",
                "recipe_name",
            ],
            ascending=[False, False, False, True, True],
        )
    elif sort_by == "fewest_missing":
        df = df.sort_values(
            by=[
                "missing_count",
                "matched_count",
                "recipe_coverage",
                "similarity",
                "recipe_name",
            ],
            ascending=[True, False, False, False, True],
        )
    elif sort_by == "lowest_calories":
        df = df.sort_values(
            by=[
                "calories_numeric",
                "missing_count",
                "matched_count",
                "similarity",
                "recipe_name",
            ],
            ascending=[True, True, False, False, True],
            na_position="last",
        )
    else:
        # default: "best_match"
        df = df.sort_values(
            by=[
                "final_score",
                "similarity",
                "matched_count",
                "recipe_coverage",
                "missing_count",
                "recipe_name",
            ],
            ascending=[False, False, False, False, True, True],
        )

    return df.reset_index(drop=True)