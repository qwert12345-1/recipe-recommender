from datasets import load_dataset
import pandas as pd
import os
import ast


def load_data():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("datahiveai/recipes-with-nutrition", split="train")
    print(f"Loaded dataset with {len(dataset)} rows")
    return dataset


def to_dataframe(dataset):
    print("Converting dataset to pandas DataFrame...")
    df = dataset.to_pandas()
    print(f"DataFrame shape: {df.shape}")
    return df


def parse_ingredients_field(x):
    """
    Convert the ingredients field into a Python list if needed.
    It may already be a list, or it may be a string representation of a list.
    """
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    return []


def extract_ingredient_names(ingredients_value):
    """
    Extract ingredient names from parsed ingredient data.
    """
    ingredients_list = parse_ingredients_field(ingredients_value)
    names = []

    for item in ingredients_list:
        if isinstance(item, dict) and "food" in item:
            name = str(item["food"]).lower().strip()
            if name:
                names.append(name)

    return names


def clean_data(df):
    print("Checking ingredient field type...")
    print("Type of first ingredients entry before parsing:", type(df["ingredients"].iloc[0]))

    print("Extracting ingredient names...")
    df["ingredient_names"] = df["ingredients"].apply(extract_ingredient_names)

    print("Sample extracted ingredient_names:")
    print(df["ingredient_names"].head())

    print("Removing invalid rows...")
    df = df[df["ingredient_names"].map(len) > 0]
    df = df.dropna(subset=["ingredient_names"])

    print(f"Remaining rows after cleaning: {len(df)}")
    return df


def main():
    os.makedirs("data/processed", exist_ok=True)

    dataset = load_data()
    df = to_dataframe(dataset)

    print("\nSample raw row:")
    print(df.iloc[0])

    df = clean_data(df)

    df = df[
        [
            "recipe_name",
            "ingredient_names",
            "calories",
            "cuisine_type",
            "diet_labels",
        ]
    ]

    output_path = "data/processed/clean_recipes.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to: {output_path}")


if __name__ == "__main__":
    main()