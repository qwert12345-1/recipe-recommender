import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from similarity import recommend_recipes, load_recipe_metadata


def flatten_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    values = set()

    if column not in df.columns:
        return []

    for entry in df[column]:
        if isinstance(entry, list):
            for item in entry:
                if item is not None and str(item).strip():
                    values.add(str(item).strip())
        elif pd.notna(entry):
            text = str(entry).strip()
            if text:
                values.add(text)

    return sorted(values)


def render_list_value(value):
    if isinstance(value, list) and value:
        return ", ".join(map(str, value))
    return "N/A"


st.set_page_config(page_title="Recipe Recommender", layout="wide")
st.title("🍽️ Recipe Recommendation System")

metadata_df = load_recipe_metadata("models/recipe_metadata.csv")

cuisine_options = flatten_unique_values(metadata_df, "cuisine_type")
meal_type_options = flatten_unique_values(metadata_df, "meal_type")
dish_type_options = flatten_unique_values(metadata_df, "dish_type")
diet_label_options = flatten_unique_values(metadata_df, "diet_labels")

st.header("Enter Ingredients")

ingredients_input = st.text_input(
    "Ingredients (comma-separated)",
    placeholder="e.g. chicken, garlic, olive oil"
)

top_k = st.slider("Number of results", 1, 20, 5)

st.header("Optional Filters")

include_cuisine = st.selectbox("Include cuisine", ["None"] + cuisine_options)
exclude_cuisines = st.multiselect("Exclude cuisines", cuisine_options)

meal_type = st.selectbox("Meal type", ["None"] + meal_type_options)
dish_type = st.selectbox("Dish type", ["None"] + dish_type_options)
diet_label = st.selectbox("Diet label", ["None"] + diet_label_options)

min_calories = st.number_input("Min calories", min_value=0.0, value=0.0)
max_calories = st.number_input("Max calories", min_value=0.0, value=10000.0)

exclude_ingredients_text = st.text_input(
    "Exclude ingredients (comma-separated)",
    placeholder="e.g. butter, peanuts"
)

min_match_count = st.slider("Minimum matched ingredients", 0, 10, 0)
pantry_only = st.checkbox("Only show recipes I can make right now", value=False)

if st.button("Find Recipes"):
    if not ingredients_input.strip():
        st.warning("Please enter at least one ingredient.")
    else:
        query_ingredients = [x.strip() for x in ingredients_input.split(",") if x.strip()]
        exclude_ingredients_list = [
            x.strip() for x in exclude_ingredients_text.split(",") if x.strip()
        ]

        selected_include_cuisine = None if include_cuisine == "None" else include_cuisine
        selected_meal_type = None if meal_type == "None" else meal_type
        selected_dish_type = None if dish_type == "None" else dish_type
        selected_diet_label = None if diet_label == "None" else diet_label

        try:
            results = recommend_recipes(
                query_ingredients=query_ingredients,
                top_k=top_k,
                include_cuisine=selected_include_cuisine,
                exclude_cuisines=exclude_cuisines or None,
                meal_type=selected_meal_type,
                dish_type=selected_dish_type,
                diet_label=selected_diet_label,
                min_calories=min_calories,
                max_calories=max_calories,
                exclude_ingredients=exclude_ingredients_list or None,
                require_pantry_subset=pantry_only,
                min_match_count=min_match_count if min_match_count > 0 else None,
            )

            if results.empty:
                st.warning("No recipes found with these filters.")
            else:
                st.success(f"Found {len(results)} recipes!")

                for _, row in results.iterrows():
                    st.markdown("---")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        image_url = row.get("image_url")
                        if pd.notna(image_url) and str(image_url).strip():
                            st.image(image_url, use_container_width=True)
                        else:
                            st.write("No image available")

                    with col2:
                        st.subheader(row["recipe_name"])

                        url = row.get("url")
                        if pd.notna(url) and str(url).strip():
                            st.markdown(f"[Open full recipe]({url})")

                        st.write(f"**Similarity:** {row['similarity']:.4f}")
                        st.write(f"**Matched Ingredients:** {render_list_value(row.get('matched_ingredients', []))}")
                        st.write(f"**Missing Ingredients:** {render_list_value(row.get('missing_ingredients', []))}")
                        st.write(f"**Missing Count:** {row.get('missing_count', 0)}")

                        calories_value = pd.to_numeric(row.get("calories"), errors="coerce")
                        if pd.notna(calories_value):
                            st.write(f"**Calories (total):** {calories_value:.0f}")
                        else:
                            st.write("**Calories (total):** N/A")

                        st.write(f"**Servings:** {row.get('servings', 'N/A')}")
                        st.write(f"**Cuisine:** {render_list_value(row.get('cuisine_type', []))}")
                        st.write(f"**Meal Type:** {render_list_value(row.get('meal_type', []))}")
                        st.write(f"**Dish Type:** {render_list_value(row.get('dish_type', []))}")
                        st.write(f"**Diet Labels:** {render_list_value(row.get('diet_labels', []))}")

                        ingredient_lines = row.get("ingredient_lines", [])
                        if isinstance(ingredient_lines, list) and ingredient_lines:
                            with st.expander("Show ingredient lines"):
                                for line in ingredient_lines:
                                    st.write(f"- {line}")

        except Exception as e:
            st.error(f"Error: {str(e)}")