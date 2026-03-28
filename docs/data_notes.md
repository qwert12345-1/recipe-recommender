# Data Processing Notes (Member 1)

## Overview
This document describes how the raw recipe dataset was cleaned and structured for use in downstream modules (vectorization, similarity, filtering, and the web app).

The dataset used is:
- `datahiveai/recipes-with-nutrition` (Hugging Face)

---

## Outputs

The preprocessing pipeline produces three files in `data/processed/`:

### 1. clean_recipes_selected.csv
Full cleaned dataset with selected columns and added normalized fields.

Includes:
- recipe_name
- url
- servings
- calories
- image_url
- diet_labels
- health_labels
- cuisine_type
- meal_type
- dish_type
- ingredient_lines
- ingredients
- total_nutrients
- daily_values
- ingredient_names (new)

---

### 2. recipes_for_model.csv
Lightweight dataset for machine learning modules.

Includes:
- recipe_name
- ingredient_names
- calories
- cuisine_type
- diet_labels
- health_labels
- meal_type
- dish_type

This file is intended for:
- vectorization
- similarity computation
- filtering

---

### 3. ingredient_vocab.csv
Frequency table of all normalized ingredients.

Columns:
- ingredient
- count

This is useful for:
- building vocabularies
- mapping ingredients to indices

---

## Key Processing Steps

### 1. Column Selection
Only relevant columns were retained to reduce noise and improve consistency.

---

### 2. Parsing List-Like Columns
Columns such as:
- diet_labels
- health_labels
- cuisine_type
- meal_type
- dish_type
- ingredients

may appear as strings in CSV format. These are parsed into Python lists using `ast.literal_eval`.

---

### 3. Ingredient Normalization

Ingredient names were extracted from the `ingredients` field using the `food` attribute.

Normalization steps:
- converted to lowercase
- removed punctuation and special characters
- stripped whitespace
- removed duplicates within each recipe

Example:
Raw: "Egg ", "EGG!", "egg"
Cleaned: ["egg"]

---

### 4. Metadata Normalization

The following columns were normalized to consistent list format:
- cuisine_type
- diet_labels
- health_labels
- meal_type
- dish_type

Each value:
- converted to lowercase
- stripped of whitespace
- ensured to be a list

Example:
["American"] → ["american"]

---

### 5. Numeric Cleaning

- `calories` converted to numeric
- rows with invalid or missing calorie values were removed

---

### 6. Row Filtering

Rows were removed if:
- no valid ingredients could be extracted
- calories were missing or invalid

---

## Data Format Notes (IMPORTANT)

Since CSV does not support Python objects:

- List columns (e.g., ingredient_names, cuisine_type) are stored as strings in the CSV
- Downstream modules should convert them back using:

```python
import ast

df["ingredient_names"] = df["ingredient_names"].apply(ast.literal_eval)