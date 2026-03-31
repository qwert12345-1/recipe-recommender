from typing import List, Optional, Set

import pandas as pd
import torch
import torch.nn.functional as F

from filtering import apply_match_constraints, build_filter_mask
from ingredient_utils import (
    normalize_ingredient_text,
    normalize_and_dedupe_ingredients,
    safe_json_loads_list,
)
from ranking import rerank_results
from vectorization import load_vocab, query_to_vector


class RecipeSimilarityError(Exception):
    """Custom exception for similarity / retrieval errors."""
    pass


def load_recipe_matrix(tensor_path: str = "models/recipe_matrix.pt") -> torch.Tensor:
    """
    Load the saved recipe matrix.
    Shape: [num_recipes, vocab_size]
    """
    recipe_matrix = torch.load(tensor_path)

    if not isinstance(recipe_matrix, torch.Tensor):
        raise RecipeSimilarityError("Loaded recipe matrix is not a torch.Tensor.")

    if recipe_matrix.ndim != 2:
        raise RecipeSimilarityError(
            f"Recipe matrix must be 2D, got shape {tuple(recipe_matrix.shape)}"
        )

    return recipe_matrix.float()


def load_recipe_metadata(metadata_path: str = "models/recipe_metadata.csv") -> pd.DataFrame:
    """
    Load metadata aligned row-for-row with the recipe matrix.
    """
    df = pd.read_csv(metadata_path)

    json_columns = [
        "ingredients_clean",
        "ingredient_lines",
        "cuisine_type",
        "diet_labels",
        "health_labels",
        "meal_type",
        "dish_type",
    ]

    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_json_loads_list)

    required_cols = ["recipe_id", "recipe_name", "ingredients_clean"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RecipeSimilarityError(f"Metadata missing required columns: {missing}")

    return df


def cosine_similarity_scores(
    query_vector: torch.Tensor,
    recipe_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between the query vector and all recipe vectors.
    """
    if query_vector.ndim != 1:
        raise RecipeSimilarityError("query_vector must be 1D.")

    if recipe_matrix.ndim != 2:
        raise RecipeSimilarityError("recipe_matrix must be 2D.")

    if query_vector.shape[0] != recipe_matrix.shape[1]:
        raise RecipeSimilarityError(
            f"Dimension mismatch: query has {query_vector.shape[0]} dims, "
            f"recipe matrix has width {recipe_matrix.shape[1]}"
        )

    query_vector = query_vector.float()
    recipe_matrix = recipe_matrix.float()

    query_norm = F.normalize(query_vector.unsqueeze(0), p=2, dim=1)
    recipe_norm = F.normalize(recipe_matrix, p=2, dim=1)

    scores = torch.matmul(recipe_norm, query_norm.squeeze(0))
    return scores


def build_recipe_ingredient_sets_from_metadata(
    metadata_df: pd.DataFrame,
) -> List[Set[str]]:
    """
    Build recipe ingredient sets directly from metadata ingredients_clean.
    This is simpler and easier to explain than reconstructing from the tensor.
    """
    recipe_ingredient_sets = []

    for ingredients in metadata_df["ingredients_clean"]:
        if isinstance(ingredients, list):
            normalized = normalize_and_dedupe_ingredients(ingredients)
            recipe_ingredient_sets.append(set(normalized))
        else:
            recipe_ingredient_sets.append(set())

    return recipe_ingredient_sets


def get_top_k_matches(
    query_ingredients: List[str],
    recipe_matrix_path: str = "models/recipe_matrix.pt",
    metadata_path: str = "models/recipe_metadata.csv",
    vocab_path: str = "models/ingredient_vocab_member2.csv",
    top_k: int = 10,
    include_cuisine: Optional[str] = None,
    exclude_cuisines: Optional[List[str]] = None,
    meal_type: Optional[str] = None,
    dish_type: Optional[str] = None,
    diet_label: Optional[str] = None,
    min_calories: Optional[float] = None,
    max_calories: Optional[float] = None,
    exclude_ingredients: Optional[List[str]] = None,
    require_all_query_matches: bool = False,
    require_pantry_subset: bool = False,
    min_match_count: Optional[int] = None,
    sort_by: str = "best_match",
) -> pd.DataFrame:
    """
    Main retrieval function.

    Stage 1: similarity / retrieval
    Stage 2: filter / constraints
    Stage 3: reranking for final display
    """
    if not query_ingredients:
        raise RecipeSimilarityError("query_ingredients cannot be empty.")

    normalized_query = normalize_and_dedupe_ingredients(query_ingredients)

    if not normalized_query:
        raise RecipeSimilarityError("No valid query ingredients after normalization.")

    ingredient_to_idx, _, _ = load_vocab(vocab_path)

    query_vec = query_to_vector(
        query_ingredients=normalized_query,
        ingredient_to_idx=ingredient_to_idx,
        vector_type="binary",
    )

    if torch.count_nonzero(query_vec).item() == 0:
        raise RecipeSimilarityError(
            "None of the query ingredients were found in the vocabulary."
        )

    recipe_matrix = load_recipe_matrix(recipe_matrix_path)
    metadata_df = load_recipe_metadata(metadata_path)

    if recipe_matrix.shape[0] != len(metadata_df):
        raise RecipeSimilarityError(
            "Row mismatch: recipe matrix rows do not match metadata rows."
        )

    scores = cosine_similarity_scores(query_vec, recipe_matrix)

    results_df = metadata_df.copy()
    results_df["similarity"] = scores.detach().cpu().numpy()

    recipe_ingredient_sets = build_recipe_ingredient_sets_from_metadata(metadata_df)

    query_set = set(normalized_query)
    matched_ingredients_col = []
    matched_count_col = []
    query_coverage_col = []
    recipe_coverage_col = []
    missing_ingredients_col = []
    missing_count_col = []
    extra_query_ingredients_col = []

    for ingredient_set in recipe_ingredient_sets:
        matched = sorted(query_set.intersection(ingredient_set))
        missing = sorted(ingredient_set.difference(query_set))
        extra_query = sorted(query_set.difference(ingredient_set))

        matched_count = len(matched)
        query_coverage = matched_count / len(query_set) if query_set else 0.0
        recipe_coverage = matched_count / len(ingredient_set) if ingredient_set else 0.0

        matched_ingredients_col.append(matched)
        matched_count_col.append(matched_count)
        query_coverage_col.append(query_coverage)
        recipe_coverage_col.append(recipe_coverage)
        missing_ingredients_col.append(missing)
        missing_count_col.append(len(missing))
        extra_query_ingredients_col.append(extra_query)

    results_df["matched_ingredients"] = matched_ingredients_col
    results_df["matched_count"] = matched_count_col
    results_df["query_coverage"] = query_coverage_col
    results_df["recipe_coverage"] = recipe_coverage_col
    results_df["missing_ingredients"] = missing_ingredients_col
    results_df["missing_count"] = missing_count_col
    results_df["extra_query_ingredients"] = extra_query_ingredients_col

    filter_mask = build_filter_mask(
        results_df=results_df,
        recipe_ingredient_sets=recipe_ingredient_sets,
        normalize_ingredient_text_func=normalize_ingredient_text,
        include_cuisine=include_cuisine,
        exclude_cuisines=exclude_cuisines,
        meal_type=meal_type,
        dish_type=dish_type,
        diet_label=diet_label,
        min_calories=min_calories,
        max_calories=max_calories,
        exclude_ingredients=exclude_ingredients,
    )

    results_df = results_df[filter_mask].copy()

    results_df = apply_match_constraints(
        results_df=results_df,
        query_size=len(query_set),
        require_all_query_matches=require_all_query_matches,
        min_match_count=min_match_count,
    )

    if require_pantry_subset:
        results_df = results_df[results_df["missing_count"] == 0].copy()

    results_df = rerank_results(results_df, sort_by=sort_by)

    return results_df.head(top_k)


def recommend_recipes(
    query_ingredients: List[str],
    top_k: int = 5,
    include_cuisine: Optional[str] = None,
    exclude_cuisines: Optional[List[str]] = None,
    meal_type: Optional[str] = None,
    dish_type: Optional[str] = None,
    diet_label: Optional[str] = None,
    min_calories: Optional[float] = None,
    max_calories: Optional[float] = None,
    exclude_ingredients: Optional[List[str]] = None,
    require_all_query_matches: bool = False,
    require_pantry_subset: bool = False,
    min_match_count: Optional[int] = None,
    sort_by: str = "best_match",
) -> pd.DataFrame:
    return get_top_k_matches(
        query_ingredients=query_ingredients,
        top_k=top_k,
        include_cuisine=include_cuisine,
        exclude_cuisines=exclude_cuisines,
        meal_type=meal_type,
        dish_type=dish_type,
        diet_label=diet_label,
        min_calories=min_calories,
        max_calories=max_calories,
        exclude_ingredients=exclude_ingredients,
        require_all_query_matches=require_all_query_matches,
        require_pantry_subset=require_pantry_subset,
        min_match_count=min_match_count,
        sort_by=sort_by,
    )


def pretty_print_results(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("No matching recipes found.")
        return

    for i, row in results_df.iterrows():
        print(f"\nRank #{i + 1}")
        print(f"Recipe ID: {row.get('recipe_id')}")
        print(f"Recipe Name: {row.get('recipe_name')}")
        print(f"Similarity: {row.get('similarity', 0):.4f}")
        print(f"Final Score: {row.get('final_score', 0):.4f}")
        print(f"Matched Count: {row.get('matched_count', 0)}")
        print(f"Query Coverage: {row.get('query_coverage', 0):.2%}")
        print(f"Recipe Coverage: {row.get('recipe_coverage', 0):.2%}")
        print(f"Matched Ingredients: {row.get('matched_ingredients', [])}")
        print(f"Calories: {row.get('calories', 'N/A')}")
        print(f"Cuisine: {row.get('cuisine_type', [])}")
        print(f"Meal Type: {row.get('meal_type', [])}")
        print(f"Dish Type: {row.get('dish_type', [])}")
        print(f"Diet Labels: {row.get('diet_labels', [])}")
        print(f"Missing Ingredients: {row.get('missing_ingredients', [])}")
        print(f"Missing Count: {row.get('missing_count', 0)}")
        print(f"Extra Query Ingredients: {row.get('extra_query_ingredients', [])}")


if __name__ == "__main__":
    print("=" * 80)
    print("SAMPLE 1: ingredient-only retrieval")
    print("=" * 80)

    results_1 = recommend_recipes(
        query_ingredients=["chicken", "garlic", "olive oil", "salt"],
        top_k=5,
        sort_by="best_match",
    )
    pretty_print_results(results_1)

    print("\n" + "=" * 80)
    print("SAMPLE 2: filtered retrieval")
    print("=" * 80)

    results_2 = recommend_recipes(
        query_ingredients=["chicken", "garlic", "olive oil", "salt"],
        top_k=5,
        include_cuisine="mediterranean",
        exclude_cuisines=["american"],
        min_calories=200,
        max_calories=6000,
        exclude_ingredients=["butter"],
        min_match_count=2,
        sort_by="fewest_missing",
    )
    pretty_print_results(results_2)