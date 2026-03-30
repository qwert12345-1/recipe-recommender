import os
import json
from typing import List, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RecipeVectorizationError(Exception):
    pass


def load_model_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load the processed modeling CSV and parse JSON-serialized list columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    df = pd.read_csv(csv_path)

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
            df[col] = df[col].apply(_safe_json_loads_list)

    required_cols = ["recipe_id", "recipe_name", "ingredients_clean"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RecipeVectorizationError(
            f"Missing required columns in modeling CSV: {missing}"
        )

    return df


def _safe_json_loads_list(x) -> List[str]:
    """
    Safely parse a JSON list. Return [] if malformed or missing.
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


def normalize_query_ingredient(text: str) -> Optional[str]:
    """
    Keep query normalization aligned with Member 1's ingredient normalization.
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

    if not text:
        return None

    return text


def build_vocab_from_dataframe(
    df: pd.DataFrame,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str], pd.DataFrame]:
    """
    Build ingredient vocabulary from ingredients_clean column.

    Returns:
        ingredient_to_idx
        idx_to_ingredient
        vocab_df
    """
    if "ingredients_clean" not in df.columns:
        raise RecipeVectorizationError("DataFrame must contain 'ingredients_clean'.")

    counter: Dict[str, int] = {}

    for ingredients in df["ingredients_clean"]:
        if not isinstance(ingredients, list):
            continue
        for ing in ingredients:
            if isinstance(ing, str) and ing:
                counter[ing] = counter.get(ing, 0) + 1

    vocab_items = [
        {"ingredient": ingredient, "count": count}
        for ingredient, count in counter.items()
        if count >= min_freq
    ]

    vocab_df = pd.DataFrame(vocab_items).sort_values(
        by=["count", "ingredient"], ascending=[False, True]
    ).reset_index(drop=True)

    if max_vocab_size is not None:
        vocab_df = vocab_df.head(max_vocab_size).copy()

    ingredient_to_idx = {
        ingredient: idx for idx, ingredient in enumerate(vocab_df["ingredient"].tolist())
    }
    idx_to_ingredient = {idx: ingredient for ingredient, idx in ingredient_to_idx.items()}

    return ingredient_to_idx, idx_to_ingredient, vocab_df


def save_vocab(vocab_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vocab_df.to_csv(save_path, index=False)


def load_vocab(vocab_csv_path: str) -> Tuple[Dict[str, int], Dict[int, str], pd.DataFrame]:
    if not os.path.exists(vocab_csv_path):
        raise FileNotFoundError(f"Could not find vocab file: {vocab_csv_path}")

    vocab_df = pd.read_csv(vocab_csv_path)

    required_cols = ["ingredient", "count"]
    missing = [col for col in required_cols if col not in vocab_df.columns]
    if missing:
        raise RecipeVectorizationError(f"Vocab file missing columns: {missing}")

    ingredient_to_idx = {
        ingredient: idx for idx, ingredient in enumerate(vocab_df["ingredient"].tolist())
    }
    idx_to_ingredient = {idx: ingredient for ingredient, idx in ingredient_to_idx.items()}

    return ingredient_to_idx, idx_to_ingredient, vocab_df


def ingredients_to_binary_vector(
    ingredients: List[str],
    ingredient_to_idx: Dict[str, int],
) -> torch.Tensor:
    """
    Convert a list of ingredients into a binary multi-hot vector.
    """
    vec = torch.zeros(len(ingredient_to_idx), dtype=torch.float32)

    if not isinstance(ingredients, list):
        return vec

    for ing in ingredients:
        if ing in ingredient_to_idx:
            vec[ingredient_to_idx[ing]] = 1.0

    return vec


def ingredients_to_count_vector(
    ingredients: List[str],
    ingredient_to_idx: Dict[str, int],
) -> torch.Tensor:
    """
    Convert a list of ingredients into a count vector.
    Usually binary is enough for your project, but this supports 'weighted' vectors too.
    """
    vec = torch.zeros(len(ingredient_to_idx), dtype=torch.float32)

    if not isinstance(ingredients, list):
        return vec

    for ing in ingredients:
        if ing in ingredient_to_idx:
            vec[ingredient_to_idx[ing]] += 1.0

    return vec


def dataframe_to_recipe_matrix(
    df: pd.DataFrame,
    ingredient_to_idx: Dict[str, int],
    vector_type: str = "binary",
) -> torch.Tensor:
    """
    Convert the whole recipe table into a recipe matrix of shape:
        [num_recipes, vocab_size]
    """
    if vector_type not in {"binary", "count"}:
        raise ValueError("vector_type must be 'binary' or 'count'")

    vectors = []

    for ingredients in df["ingredients_clean"]:
        if vector_type == "binary":
            vec = ingredients_to_binary_vector(ingredients, ingredient_to_idx)
        else:
            vec = ingredients_to_count_vector(ingredients, ingredient_to_idx)
        vectors.append(vec)

    if not vectors:
        return torch.empty((0, len(ingredient_to_idx)), dtype=torch.float32)

    return torch.stack(vectors, dim=0)


def query_to_vector(
    query_ingredients: List[str],
    ingredient_to_idx: Dict[str, int],
    vector_type: str = "binary",
) -> torch.Tensor:
    """
    Convert user input ingredients into the same vector format as recipes.
    """
    normalized = []
    for ing in query_ingredients:
        norm = normalize_query_ingredient(ing)
        if norm:
            normalized.append(norm)

    if vector_type == "binary":
        return ingredients_to_binary_vector(normalized, ingredient_to_idx)
    elif vector_type == "count":
        return ingredients_to_count_vector(normalized, ingredient_to_idx)
    else:
        raise ValueError("vector_type must be 'binary' or 'count'")


def save_tensor(tensor: torch.Tensor, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor, save_path)


def load_tensor(load_path: str) -> torch.Tensor:
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Could not find tensor file: {load_path}")
    return torch.load(load_path)


def save_recipe_metadata(df: pd.DataFrame, save_path: str) -> None:
    """
    Save metadata aligned row-for-row with the recipe tensor matrix.
    Keep enough fields for rich Streamlit recipe cards.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metadata_cols = [
        col for col in [
            "recipe_id",
            "recipe_name",
            "url",
            "image_url",
            "servings",
            "ingredient_lines",
            "ingredients_clean",
            "calories",
            "cuisine_type",
            "diet_labels",
            "health_labels",
            "meal_type",
            "dish_type",
        ]
        if col in df.columns
    ]

    metadata_df = df[metadata_cols].copy()

    for col in metadata_df.columns:
        if metadata_df[col].apply(lambda x: isinstance(x, list)).any():
            metadata_df[col] = metadata_df[col].apply(json.dumps)

    metadata_df.to_csv(save_path, index=False)


class RecipeTensorDataset(Dataset):
    """
    Small PyTorch Dataset wrapper so Member 3 can use batching easily.
    """

    def __init__(self, recipe_matrix: torch.Tensor):
        if recipe_matrix.ndim != 2:
            raise ValueError("recipe_matrix must be 2D: [num_recipes, vocab_size]")
        self.recipe_matrix = recipe_matrix

    def __len__(self) -> int:
        return self.recipe_matrix.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.recipe_matrix[idx]


def build_recipe_dataloader(
    recipe_matrix: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    dataset = RecipeTensorDataset(recipe_matrix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_vectorization_pipeline(
    model_csv_path: str = "data/processed/recipes_for_model.csv",
    vocab_save_path: str = "models/ingredient_vocab_member2.csv",
    tensor_save_path: str = "models/recipe_matrix.pt",
    metadata_save_path: str = "models/recipe_metadata.csv",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    vector_type: str = "binary",
) -> None:
    """
    End-to-end Member 2 pipeline:
    1. load processed data
    2. build vocab
    3. vectorize recipes
    4. save vocab, tensor, metadata
    """
    print("Loading processed modeling CSV...")
    df = load_model_dataframe(model_csv_path)
    print(f"Loaded {len(df)} recipes")

    print("Building ingredient vocabulary...")
    ingredient_to_idx, idx_to_ingredient, vocab_df = build_vocab_from_dataframe(
        df,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )
    print(f"Vocabulary size: {len(ingredient_to_idx)}")

    print(f"Building recipe matrix using vector_type='{vector_type}'...")
    recipe_matrix = dataframe_to_recipe_matrix(
        df,
        ingredient_to_idx,
        vector_type=vector_type,
    )
    print(f"Recipe matrix shape: {tuple(recipe_matrix.shape)}")

    print("Saving outputs...")
    save_vocab(vocab_df, vocab_save_path)
    save_tensor(recipe_matrix, tensor_save_path)
    save_recipe_metadata(df, metadata_save_path)

    print(f"Saved vocab to: {vocab_save_path}")
    print(f"Saved recipe matrix to: {tensor_save_path}")
    print(f"Saved recipe metadata to: {metadata_save_path}")

    print("Done.")


def demo_query_vectorization(
    query_ingredients: List[str],
    vocab_csv_path: str = "models/ingredient_vocab_member2.csv",
    vector_type: str = "binary",
) -> torch.Tensor:
    """
    Convenience helper for quick testing.
    """
    ingredient_to_idx, _, _ = load_vocab(vocab_csv_path)
    q = query_to_vector(query_ingredients, ingredient_to_idx, vector_type=vector_type)
    return q


if __name__ == "__main__":
    run_vectorization_pipeline()