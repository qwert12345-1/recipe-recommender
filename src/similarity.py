import json
from typing import List, Optional, Set

import pandas as pd
import torch
import torch.nn.functional as F

from filtering import apply_match_constraints, build_filter_mask
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


def _safe_json_loads_list(x):
    """
    Safely parse JSON list columns from recipe_metadata.csv.
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def load_recipe_metadata(metadata_path: str = "models/recipe_metadata.csv") -> pd.DataFrame:
    """
    Load metadata aligned row-for-row with the recipe matrix.
    """
    df = pd.read_csv(metadata_path)

    json_columns = [
        "cuisine_type",
        "diet_labels",
        "health_labels",
        "meal_type",
        "dish_type",
    ]

    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(_safe_json_loads_list)

    required_cols = ["recipe_id", "recipe_name"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RecipeSimilarityError(f"Metadata missing required columns: {missing}")

    return df


def normalize_ingredient_text(text: str) -> Optional[str]:
    """
    Normalize user-entered ingredient text to match the preprocessing style.
    """
    if not isinstance(text, str):
        return None

    text = text.lower().strip()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")

    cleaned = []
    for ch in text:
        if ch.isalpha() or ch.isspace():
            cleaned.append(ch)

    text = "".join(cleaned)
    text = " ".join(text.split())

    return text if text else None


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


def reconstruct_recipe_ingredient_sets(
    recipe_matrix: torch.Tensor,
    vocab_path: str,
) -> List[Set[str]]:
    """
    Reconstruct each recipe's ingredient set from the recipe matrix + vocab.
    This lets us support exclude_ingredients and display matched ingredients.
    """
    _, idx_to_ingredient, _ = load_vocab(vocab_path)
    binary_matrix = recipe_matrix > 0

    ingredient_sets = []
    for row_idx in range(binary_matrix.shape[0]):
        active_indices = torch.nonzero(
            binary_matrix[row_idx], as_tuple=False
        ).flatten().tolist()
        ingredient_set = {idx_to_ingredient[i] for i in active_indices}
        ingredient_sets.append(ingredient_set)

    return ingredient_sets


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
    min_match_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Main retrieval function.

    Ranking is based primarily on cosine similarity to the user ingredient query.
    Filtering / constraints are applied after similarity scores are computed.
    """
    if not query_ingredients:
        raise RecipeSimilarityError("query_ingredients cannot be empty.")

    normalized_query = []
    for ing in query_ingredients:
        norm = normalize_ingredient_text(ing)
        if norm:
            normalized_query.append(norm)

    normalized_query = list(dict.fromkeys(normalized_query))

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

    recipe_ingredient_sets = reconstruct_recipe_ingredient_sets(recipe_matrix, vocab_path)

    query_set = set(normalized_query)
    matched_ingredients_col = []
    matched_count_col = []
    query_coverage_col = []

    for ingredient_set in recipe_ingredient_sets:
        matched = sorted(query_set.intersection(ingredient_set))
        matched_count = len(matched)
        coverage = matched_count / len(query_set) if query_set else 0.0

        matched_ingredients_col.append(matched)
        matched_count_col.append(matched_count)
        query_coverage_col.append(coverage)

    results_df["matched_ingredients"] = matched_ingredients_col
    results_df["matched_count"] = matched_count_col
    results_df["query_coverage"] = query_coverage_col

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

    if "calories" in results_df.columns:
        results_df["calories_numeric"] = pd.to_numeric(results_df["calories"], errors="coerce")
    else:
        results_df["calories_numeric"] = float("inf")

    results_df = results_df.sort_values(
        by=["similarity", "matched_count", "query_coverage", "calories_numeric", "recipe_name"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    results_df = results_df.drop(columns=["calories_numeric"])

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
    min_match_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for the main retrieval function.
    """
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
        min_match_count=min_match_count,
    )


def pretty_print_results(results_df: pd.DataFrame) -> None:
    """
    Print results in a demo-friendly format.
    """
    if results_df.empty:
        print("No matching recipes found.")
        return

    for i, row in results_df.iterrows():
        print(f"\nRank #{i + 1}")
        print(f"Recipe ID: {row.get('recipe_id')}")
        print(f"Recipe Name: {row.get('recipe_name')}")
        print(f"Similarity: {row.get('similarity', 0):.4f}")
        print(f"Matched Count: {row.get('matched_count', 0)}")
        print(f"Query Coverage: {row.get('query_coverage', 0):.2%}")
        print(f"Matched Ingredients: {row.get('matched_ingredients', [])}")
        print(f"Calories: {row.get('calories', 'N/A')}")
        print(f"Cuisine: {row.get('cuisine_type', [])}")
        print(f"Meal Type: {row.get('meal_type', [])}")
        print(f"Dish Type: {row.get('dish_type', [])}")
        print(f"Diet Labels: {row.get('diet_labels', [])}")


if __name__ == "__main__":
    print("=" * 80)
    print("SAMPLE 1: ingredient-only retrieval")
    print("=" * 80)

    results_1 = recommend_recipes(
        query_ingredients=["chicken", "garlic", "olive oil", "salt"],
        top_k=5,
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
    )
    pretty_print_results(results_2)