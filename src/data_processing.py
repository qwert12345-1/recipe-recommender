from datasets import load_dataset
import pandas as pd
import os
import ast
import re
import json
from collections import Counter

COLUMNS_TO_KEEP = [
    "recipe_name",
    "url",
    "servings",
    "calories",
    "image_url",
    "diet_labels",
    "health_labels",
    "cuisine_type",
    "meal_type",
    "dish_type",
    "ingredient_lines",
    "ingredients",
    "total_nutrients",
    "daily_values",
]

LIST_LIKE_COLUMNS = [
    "diet_labels",
    "health_labels",
    "cuisine_type",
    "meal_type",
    "dish_type",
    "ingredient_lines",
    "ingredients",
    "total_nutrients",
    "daily_values",
]


def load_data():
    """
    Load the Hugging Face dataset.
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("datahiveai/recipes-with-nutrition", split="train")
    print(f"Loaded dataset with {len(dataset)} rows")
    return dataset


def to_dataframe(dataset):
    """
    Convert Hugging Face dataset to pandas DataFrame.
    """
    print("Converting dataset to pandas DataFrame...")
    df = dataset.to_pandas()
    print(f"Original DataFrame shape: {df.shape}")
    return df


def parse_maybe_literal(x):
    """
    Convert a string representation of a list/dict into a Python object.
    If x is already a list/dict, return it unchanged.
    """
    if isinstance(x, (list, dict)):
        return x

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    return x


def safe_json_dumps(x):
    """
    Save list/dict columns as JSON strings so downstream teammates
    can reliably recover them with json.loads().
    """
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return json.dumps([])


def normalize_text_basic(text):
    """
    Lowercase, trim, and collapse internal whitespace.
    """
    if not isinstance(text, str):
        return None
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else None


def normalize_ingredient_name(name):
    """
    Normalize ingredient names for consistent downstream use.
    This is intentionally simple and conservative.
    """
    if not isinstance(name, str):
        return None

    name = name.lower().strip()

    # normalize separators
    name = name.replace("-", " ")
    name = name.replace("_", " ")
    name = name.replace("/", " ")

    # remove punctuation except spaces
    name = re.sub(r"[^a-z\s]", "", name)

    # collapse spaces
    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        return None

    return name


def normalize_list_column(x):
    """
    Normalize metadata columns that should behave like lists.
    Example:
    ['American'] -> ['american']
    """
    x = parse_maybe_literal(x)

    if not isinstance(x, list):
        return []

    normalized = []
    for item in x:
        item_str = normalize_text_basic(str(item))
        if item_str:
            normalized.append(item_str)

    return normalized


def parse_list_like_columns(df, column_names):
    """
    Parse columns that may contain list-like strings into Python objects.
    """
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].apply(parse_maybe_literal)
    return df


def extract_ingredient_names(ingredients_value):
    """
    Extract a clean list of normalized ingredient names from the raw
    ingredients field.

    Expected raw format is usually a list of dicts, where each dict may
    contain a 'food' key.
    """
    ingredients_value = parse_maybe_literal(ingredients_value)

    if not isinstance(ingredients_value, list):
        return []

    cleaned_names = []

    for item in ingredients_value:
        raw_name = None

        if isinstance(item, dict):
            # most expected field
            if "food" in item:
                raw_name = item["food"]
            # backup possibilities just in case
            elif "ingredient" in item:
                raw_name = item["ingredient"]
            elif "text" in item:
                raw_name = item["text"]

        elif isinstance(item, str):
            raw_name = item

        normalized_name = normalize_ingredient_name(raw_name)
        if normalized_name:
            cleaned_names.append(normalized_name)

    # remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in cleaned_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    return unique_names


def clean_numeric_column(series):
    """
    Convert a column to numeric where possible.
    """
    return pd.to_numeric(series, errors="coerce")


def clean_recipe_name(series):
    """
    Normalize recipe names for deduplication/display.
    """
    return (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def clean_data(df):
    """
    Main cleaning pipeline for Member 1.
    Produces a canonical ingredients_clean column for downstream use.
    """
    print("Keeping selected columns...")
    df = df[COLUMNS_TO_KEEP].copy()

    print("Parsing list-like columns...")
    df = parse_list_like_columns(df, LIST_LIKE_COLUMNS)

    print("Normalizing metadata list columns...")
    for col in ["diet_labels", "health_labels", "cuisine_type", "meal_type", "dish_type"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_list_column)

    print("Cleaning recipe names...")
    df["recipe_name"] = clean_recipe_name(df["recipe_name"])

    print("Ensuring numeric columns are numeric...")
    if "calories" in df.columns:
        df["calories"] = clean_numeric_column(df["calories"])
    if "servings" in df.columns:
        df["servings"] = clean_numeric_column(df["servings"])

    print("Creating normalized ingredient columns...")
    df["ingredient_names"] = df["ingredients"].apply(extract_ingredient_names)

    # Canonical field to be used by Member 2 / Member 3
    df["ingredients_clean"] = df["ingredient_names"]

    print("Dropping bad rows...")
    df = df.dropna(subset=["recipe_name", "calories"]).copy()
    df = df[df["ingredients_clean"].map(len) > 0].copy()

    print("Dropping duplicate recipes...")
    # Strong duplicate key using recipe name + normalized ingredients
    df["dedupe_key"] = df.apply(
        lambda row: (
            str(row["recipe_name"]).strip().lower(),
            tuple(row["ingredients_clean"]),
        ),
        axis=1,
    )
    df = df.drop_duplicates(subset=["dedupe_key"]).copy()
    df = df.drop(columns=["dedupe_key"])

    print("Resetting index and adding recipe_id...")
    df = df.reset_index(drop=True)
    df["recipe_id"] = range(len(df))

    # Put recipe_id first
    first_cols = ["recipe_id", "recipe_name"]
    remaining_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + remaining_cols]

    print(f"Shape after cleaning: {df.shape}")
    print("Sample ingredients_clean:")
    print(df["ingredients_clean"].head())

    return df


def build_ingredient_vocab(df):
    """
    Build a frequency table of normalized ingredients from ingredients_clean.
    """
    counter = Counter()

    for ingredient_list in df["ingredients_clean"]:
        counter.update(ingredient_list)

    vocab_df = pd.DataFrame(
        [{"ingredient": ingredient, "count": count} for ingredient, count in counter.items()]
    )

    vocab_df = vocab_df.sort_values("count", ascending=False).reset_index(drop=True)
    return vocab_df


def make_model_df(df):
    """
    Create a smaller downstream-friendly table for vectorization/ranking.
    """
    model_columns = [
        "recipe_id",
        "recipe_name",
        "ingredients_clean",
        "calories",
        "cuisine_type",
        "diet_labels",
        "health_labels",
        "meal_type",
        "dish_type",
    ]
    return df[model_columns].copy()


def serialize_for_csv(df):
    """
    Convert list/dict columns into JSON strings before saving CSV.
    This avoids ambiguous Python-list string formatting and makes
    downstream loading much safer.
    """
    df = df.copy()

    json_columns = [
        "diet_labels",
        "health_labels",
        "cuisine_type",
        "meal_type",
        "dish_type",
        "ingredient_lines",
        "ingredients",
        "ingredient_names",
        "ingredients_clean",
        "total_nutrients",
        "daily_values",
    ]

    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_json_dumps)

    return df


def save_outputs(df):
    """
    Save all outputs for downstream teammates.
    """
    os.makedirs("data/processed", exist_ok=True)

    # Full cleaned dataset
    full_output_path = "data/processed/clean_recipes_selected.csv"
    full_save_df = serialize_for_csv(df)
    full_save_df.to_csv(full_output_path, index=False)
    print(f"Saved full cleaned dataset to: {full_output_path}")

    # Lighter modeling dataset
    model_output_path = "data/processed/recipes_for_model.csv"
    model_df = make_model_df(df)
    model_save_df = serialize_for_csv(model_df)
    model_save_df.to_csv(model_output_path, index=False)
    print(f"Saved modeling dataset to: {model_output_path}")

    # Ingredient vocabulary
    vocab_output_path = "data/processed/ingredient_vocab.csv"
    vocab_df = build_ingredient_vocab(df)
    vocab_df.to_csv(vocab_output_path, index=False)
    print(f"Saved ingredient vocabulary to: {vocab_output_path}")


def main():
    dataset = load_data()
    df = to_dataframe(dataset)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nCleaning data...")
    df = clean_data(df)

    save_outputs(df)

    print("\nDone.")


if __name__ == "__main__":
    main()