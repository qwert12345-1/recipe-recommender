import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from similarity import recommend_recipes, load_recipe_metadata


# ----------------------------
# Session state
# ----------------------------
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

if "results" not in st.session_state:
    st.session_state.results = None

if "ingredient_rows" not in st.session_state:
    st.session_state.ingredient_rows = [
        {"value": "", "must_have": False}
    ]

# ----------------------------
# Helper functions
# ----------------------------
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

def compute_calories_per_serving(calories, servings):
    calories_num = pd.to_numeric(calories, errors="coerce")
    servings_num = pd.to_numeric(servings, errors="coerce")

    if pd.isna(calories_num) or pd.isna(servings_num) or servings_num <= 0:
        return pd.NA

    return calories_num / servings_num

def save_bookmark(row) -> None:
    recipe_id = row["recipe_id"]

    already_saved = any(
        bookmarked_recipe["recipe_id"] == recipe_id
        for bookmarked_recipe in st.session_state.bookmarks
    )

    if not already_saved:
        st.session_state.bookmarks.append(row.to_dict())


def remove_bookmark(recipe_id) -> None:
    st.session_state.bookmarks = [
        recipe for recipe in st.session_state.bookmarks
        if recipe["recipe_id"] != recipe_id
    ]


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Recipe Recommender", layout="wide")


metadata_df = load_recipe_metadata("models/recipe_metadata.csv")

cuisine_options = flatten_unique_values(metadata_df, "cuisine_type")
meal_type_options = flatten_unique_values(metadata_df, "meal_type")
dish_type_options = flatten_unique_values(metadata_df, "dish_type")
diet_label_options = flatten_unique_values(metadata_df, "diet_labels")


# ----------------------------
# Sidebar bookmarks
# ----------------------------
st.sidebar.title("Recipe App")

page = st.sidebar.radio(
    "Go to",
    ["Recipe Search", "Bookmarks", "PCA Visualization", "About"]
)

if page == "Recipe Search":
    st.title("🍽️ Recipe Recommendation System")
    st.header("Enter Ingredients")

    for i, row in enumerate(st.session_state.ingredient_rows):
        col1, col2 = st.columns([4, 1])

        with col1:
            row["value"] = st.text_input(
                f"Ingredient {i+1}",
                value=row["value"],
                key=f"ingredient_value_{i}"
            )

        with col2:
            row["must_have"] = st.checkbox(
                "Must-have",
                value=row["must_have"],
                key=f"ingredient_must_have_{i}"
            )

    col_add, col_clear = st.columns([1, 1])

    with col_add:
        if st.button("＋ Add Ingredient"):
            st.session_state.ingredient_rows.append(
                {"value": "", "must_have": False}
            )
            st.rerun()

    with col_clear:
        if st.button("Reset Ingredients"):
            st.session_state.ingredient_rows = [
                {"value": "", "must_have": False}
            ]
            st.rerun()

    top_k = st.slider("Number of results", 1, 20, 5)

    # st.header("Optional Filters")

    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            include_cuisine = st.selectbox("Include cuisine", ["None"] + cuisine_options)
            meal_type = st.selectbox("Meal type", ["None"] + meal_type_options)
            min_calories_per_serving = st.number_input(
                "Min cal / serving",
                min_value=0.0,
                value=0.0
            )
            min_servings = st.number_input(
                "Min servings",
                min_value=0.0,
                value=0.0
            )

        with col2:
            dish_type = st.selectbox("Dish type", ["None"] + dish_type_options)
            diet_label = st.selectbox("Diet label", ["None"] + diet_label_options)
            max_calories_per_serving = st.number_input(
                "Max cal / serving",
                min_value=0.0,
                value=2000.0
            )
            max_servings = st.number_input(
                "Max servings",
                min_value=0.0,
                value=20.0
            )

        with col3:
            exclude_cuisines = st.multiselect("Exclude cuisines", cuisine_options)
            pantry_only = st.checkbox("Pantry only", value=False)

        exclude_ingredients_text = st.text_input(
            "Exclude ingredients (comma-separated)",
            placeholder="e.g. butter, peanuts"
        )

    # ----------------------------
    # Search button
    # ----------------------------
    if st.button("Find Recipes"):

        query_ingredients = [
            row["value"].strip()
            for row in st.session_state.ingredient_rows
            if row["value"].strip()
        ]

        must_have_ingredients = [
            row["value"].strip()
            for row in st.session_state.ingredient_rows
            if row["value"].strip() and row["must_have"]
        ]

        if not query_ingredients:
            st.warning("Please enter at least one ingredient.")
            st.session_state.results = None
        else:
            exclude_ingredients_list = [
                x.strip() for x in exclude_ingredients_text.split(",") if x.strip()
            ]

            selected_include_cuisine = None if include_cuisine == "None" else include_cuisine
            selected_meal_type = None if meal_type == "None" else meal_type
            selected_dish_type = None if dish_type == "None" else dish_type
            selected_diet_label = None if diet_label == "None" else diet_label

            try:
                candidate_k = max(top_k * 10, 50)

                results = recommend_recipes(
                    query_ingredients=query_ingredients,
                    top_k=candidate_k,
                    include_cuisine=selected_include_cuisine,
                    exclude_cuisines=exclude_cuisines or None,
                    meal_type=selected_meal_type,
                    dish_type=selected_dish_type,
                    diet_label=selected_diet_label,
                    min_calories=None,
                    max_calories=None,
                    exclude_ingredients=exclude_ingredients_list or None,
                    require_pantry_subset=pantry_only,
                )

                # numeric cleanup
                results["calories"] = pd.to_numeric(results["calories"], errors="coerce")
                results["servings"] = pd.to_numeric(results["servings"], errors="coerce")

                # compute calories per serving
                results["calories_per_serving"] = results.apply(
                    lambda row: compute_calories_per_serving(row.get("calories"), row.get("servings")),
                    axis=1
                )

                # must-have filter
                if must_have_ingredients:
                    must_have_set = {x.lower() for x in must_have_ingredients}

                    results = results[
                        results["matched_ingredients"].apply(
                            lambda x: must_have_set.issubset(set(map(str.lower, x)))
                            if isinstance(x, list) else False
                        )
                    ].reset_index(drop=True)

                # servings filter
                if min_servings > 0:
                    results = results[results["servings"] >= min_servings]

                if max_servings > 0:
                    results = results[results["servings"] <= max_servings]

                # calories per serving filter
                if min_calories_per_serving > 0:
                    results = results[
                        results["calories_per_serving"] >= min_calories_per_serving
                    ]

                if max_calories_per_serving > 0:
                    results = results[
                        results["calories_per_serving"] <= max_calories_per_serving
                    ]

                results = results.reset_index(drop=True).head(top_k)

                st.session_state.results = results

            except Exception as e:
                st.session_state.results = None
                st.error(f"Error: {str(e)}")


    # ----------------------------
    # Render results
    # ----------------------------
    if st.session_state.results is not None:
        results = st.session_state.results

        if results.empty:
            st.warning("No recipes found with these filters.")
        else:
            st.success(f"Found {len(results)} recipes!")

            for _, row in results.iterrows():
                calories_value = pd.to_numeric(row.get("calories"), errors="coerce")
                servings_value = pd.to_numeric(row.get("servings"), errors="coerce")
                calories_per_serving_value = pd.to_numeric(
                    row.get("calories_per_serving"),
                    errors="coerce"
                )
                missing_count = row.get("missing_count", 0)

                if pd.notna(calories_per_serving_value):
                    expander_title = (
                        f"{row['recipe_name']} • 🧂 {missing_count} missing • 🍽️ {calories_per_serving_value:.0f} cal/serving"
                    )
                else:
                    expander_title = (
                        f"{row['recipe_name']} • 🧂 {missing_count} missing"
                    )

                with st.expander(expander_title, expanded=False):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        image_url = row.get("image_url")
                        if pd.notna(image_url) and str(image_url).strip():
                            st.image(image_url, use_container_width=True)
                        else:
                            st.write("No image available")

                    with col2:
                        url = row.get("url")
                        if pd.notna(url) and str(url).strip():
                            st.markdown(f"[Open full recipe]({url})")

                        st.write(f"**Similarity:** {row['similarity']:.4f}")
                        st.write(
                            f"**Matched Ingredients:** "
                            f"{render_list_value(row.get('matched_ingredients', []))}"
                        )
                        st.write(
                            f"**Missing Ingredients:** "
                            f"{render_list_value(row.get('missing_ingredients', []))}"
                        )
                        st.write(f"**Missing Count:** {missing_count}")

                        if missing_count == 0:
                            st.success("You can make this now ✅")

                        if pd.notna(calories_value):
                            st.write(f"**Calories (total):** {calories_value:.0f}")
                        else:
                            st.write("**Calories (total):** N/A")

                        if pd.notna(servings_value):
                            st.write(f"**Servings:** {servings_value:.1f}")
                        else:
                            st.write("**Servings:** N/A")

                        if pd.notna(calories_per_serving_value):
                            st.write(f"**Calories per serving:** {calories_per_serving_value:.0f}")
                        else:
                            st.write("**Calories per serving:** N/A")

                        st.write(f"**Cuisine:** {render_list_value(row.get('cuisine_type', []))}")
                        st.write(f"**Meal Type:** {render_list_value(row.get('meal_type', []))}")
                        st.write(f"**Dish Type:** {render_list_value(row.get('dish_type', []))}")
                        st.write(f"**Diet Labels:** {render_list_value(row.get('diet_labels', []))}")

                        ingredient_lines = row.get("ingredient_lines", [])
                        if isinstance(ingredient_lines, list) and ingredient_lines:
                            st.markdown("**Ingredient lines:**")
                            for line in ingredient_lines:
                                st.write(f"- {line}")

                        already_saved = any(
                            bookmarked_recipe["recipe_id"] == row["recipe_id"]
                            for bookmarked_recipe in st.session_state.bookmarks
                        )

                        if already_saved:
                            st.success("Saved in bookmarks")
                        else:
                            if st.button("⭐ Save Recipe", key=f"save_{row['recipe_id']}"):
                                save_bookmark(row)
                                st.rerun()

elif page == "Bookmarks":
    st.title("📌 Bookmarked Recipes")

    if len(st.session_state.bookmarks) == 0:
        st.write("No bookmarks yet.")
    else:
        for recipe in st.session_state.bookmarks:
            with st.expander(recipe["recipe_name"], expanded=False):
                col1, col2 = st.columns([0.8, 2.4])

                with col1:
                    image_url = recipe.get("image_url")
                    if pd.notna(image_url) and str(image_url).strip():
                        st.image(image_url, use_container_width=True)
                    else:
                        st.write("No image available")

                with col2:
                    if recipe.get("url") and str(recipe.get("url")).strip():
                        st.markdown(f"[Open recipe]({recipe['url']})")

                    calories_value = pd.to_numeric(recipe.get("calories"), errors="coerce")
                    servings_value = pd.to_numeric(recipe.get("servings"), errors="coerce")
                    calories_per_serving_value = compute_calories_per_serving(
                        recipe.get("calories"),
                        recipe.get("servings")
                    )

                    if pd.notna(calories_value):
                        st.write(f"**Calories (total):** {calories_value:.0f}")
                    else:
                        st.write("**Calories (total):** N/A")

                    if pd.notna(servings_value):
                        st.write(f"**Servings:** {servings_value:.1f}")
                    else:
                        st.write("**Servings:** N/A")

                    if pd.notna(calories_per_serving_value):
                        st.write(f"**Calories per serving:** {calories_per_serving_value:.0f}")
                    else:
                        st.write("**Calories per serving:** N/A")

                    st.write(f"**Cuisine:** {render_list_value(recipe.get('cuisine_type', []))}")
                    st.write(f"**Meal Type:** {render_list_value(recipe.get('meal_type', []))}")
                    st.write(f"**Dish Type:** {render_list_value(recipe.get('dish_type', []))}")
                    st.write(f"**Diet Labels:** {render_list_value(recipe.get('diet_labels', []))}")

                    ingredient_lines = recipe.get("ingredient_lines", [])
                    if isinstance(ingredient_lines, list) and ingredient_lines:
                        st.markdown("**Ingredient lines:**")
                        for line in ingredient_lines:
                            st.write(f"- {line}")

                    if st.button("Remove", key=f"remove_bookmark_page_{recipe['recipe_id']}"):
                        remove_bookmark(recipe["recipe_id"])
                        st.rerun()

elif page == "PCA Visualization":
    st.header("PCA Visualization")
    st.info("PCA visualization will go here.")

elif page == "About":
    st.header("About This App")


# ----------------------------
# Inputs
# ----------------------------

