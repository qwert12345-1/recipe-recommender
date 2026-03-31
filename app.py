import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import streamlit as st
import pandas as pd
from similarity import recommend_recipes, load_recipe_metadata
from pantry import PANTRY_CATEGORIES
from ingredient_utils import normalize_ingredient_text, dedupe_preserve_order

# ----------------------------
# Session state
# ----------------------------
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

if "results" not in st.session_state:
    st.session_state.results = None

if "ingredient_rows" not in st.session_state:
    st.session_state.ingredient_rows = [{"value": "", "must_have": False}]

if "ingredient_widget_version" not in st.session_state:
    st.session_state.ingredient_widget_version = 0

if "pantry_widget_version" not in st.session_state:
    st.session_state.pantry_widget_version = 0

if "filter_widget_version" not in st.session_state:
    st.session_state.filter_widget_version = 0

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

def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        cleaned = str(item).strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


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
# Sidebar
# ----------------------------
st.sidebar.title("Recipe App")

page = st.sidebar.radio(
    "Go to",
    ["Recipe Search", "Bookmarks", "PCA Visualization", "About"]
)

if page == "Recipe Search":
    st.title("🍽️ Recipe Recommendation System")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("Enter Ingredients")

        for i, row in enumerate(st.session_state.ingredient_rows):
            col1, col2 = st.columns([6, 2])

            with col1:
                row["value"] = st.text_input(
                    f"Ingredient {i + 1}",
                    value=row["value"],
                    key=f"ingredient_value_{st.session_state.ingredient_widget_version}_{i}",
                )

            with col2:
                st.markdown("<div style='margin-top: 35px'></div>", unsafe_allow_html=True)

                row["must_have"] = st.checkbox(
                    "Must-have",
                    value=row["must_have"],
                    key=f"ingredient_must_have_{st.session_state.ingredient_widget_version}_{i}",
                )

        col_add, col_reset = st.columns([1, 1])

        with col_add:
            if st.button("＋ Add Ingredient"):
                st.session_state.ingredient_rows.append({"value": "", "must_have": False})
                st.rerun()

        with col_reset:
            if st.button("Reset Ingredients"):
                st.session_state.ingredient_rows = [{"value": "", "must_have": False}]
                st.session_state.ingredient_widget_version += 1
                st.session_state.results = None
                st.rerun()

    with right_col:
        st.header("🧂 Pantry")

        selected_pantry = []

        for category, items in PANTRY_CATEGORIES.items():
            with st.expander(category, expanded=False):
                checkbox_cols = st.columns(3)

                for idx, item in enumerate(items):
                    with checkbox_cols[idx % 3]:
                        checked = st.checkbox(
                            item,
                            key=f"pantry_{st.session_state.pantry_widget_version}_{category}_{item}"
                        )
                        if checked:
                            selected_pantry.append(item)

        if st.button("Clear Pantry"):
            st.session_state.pantry_widget_version += 1
            st.session_state.results = None
            st.rerun()

    top_k = st.slider("Number of results", 1, 20, 5)

    sort_label_to_value = {
        "Best match": "best_match",
        "Highest similarity": "highest_similarity",
        "Fewest missing ingredients": "fewest_missing",
        "Lowest calories": "lowest_calories",
    }

    sort_label = st.selectbox(
        "Sort results by",
        list(sort_label_to_value.keys()),
        index=0,
    )
    sort_by = sort_label_to_value[sort_label]

    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            include_cuisine = st.selectbox(
                "Include cuisine",
                ["None"] + cuisine_options,
                key=f"include_cuisine_{st.session_state.filter_widget_version}",
            )
            meal_type = st.selectbox(
                "Meal type",
                ["None"] + meal_type_options,
                key=f"meal_type_{st.session_state.filter_widget_version}",
            )
            min_calories_per_serving = st.number_input(
                "Min cal / serving",
                min_value=0.0,
                value=0.0,
                key=f"min_calories_per_serving_{st.session_state.filter_widget_version}",
            )
            min_servings = st.number_input(
                "Min servings",
                min_value=0.0,
                value=0.0,
                key=f"min_servings_{st.session_state.filter_widget_version}",
            )

        with col2:
            dish_type = st.selectbox(
                "Dish type",
                ["None"] + dish_type_options,
                key=f"dish_type_{st.session_state.filter_widget_version}",
            )
            diet_label = st.selectbox(
                "Diet label",
                ["None"] + diet_label_options,
                key=f"diet_label_{st.session_state.filter_widget_version}",
            )
            max_calories_per_serving = st.number_input(
                "Max cal / serving",
                min_value=0.0,
                value=2000.0,
                key=f"max_calories_per_serving_{st.session_state.filter_widget_version}",
            )
            max_servings = st.number_input(
                "Max servings",
                min_value=0.0,
                value=20.0,
                key=f"max_servings_{st.session_state.filter_widget_version}",
            )

        with col3:
            exclude_cuisines = st.multiselect(
                "Exclude cuisines",
                cuisine_options,
                key=f"exclude_cuisines_{st.session_state.filter_widget_version}",
            )
            pantry_only = st.checkbox(
                "Pantry only",
                value=False,
                key=f"pantry_only_{st.session_state.filter_widget_version}",
            )
            min_match_count = st.number_input(
                "Minimum matched ingredients",
                min_value=0,
                value=0,
                step=1,
                key=f"min_match_count_{st.session_state.filter_widget_version}",
            )

        exclude_ingredients_text = st.text_input(
            "Exclude ingredients (comma-separated)",
            placeholder="e.g. butter, peanuts",
            key=f"exclude_ingredients_text_{st.session_state.filter_widget_version}",
        )
    if st.button("Clear Filters"):
        st.session_state.filter_widget_version += 1
        st.session_state.results = None
        st.rerun()

    # ----------------------------
    # Search button
    # ----------------------------
    if st.button("Find Recipes"):
        user_ingredients = [
            normalize_ingredient_text(row["value"])
            for row in st.session_state.ingredient_rows
            if row["value"].strip()
        ]

        user_ingredients = [x for x in user_ingredients if x]

        normalized_pantry = [
            normalize_ingredient_text(item)
            for item in selected_pantry
        ]

        normalized_pantry = [x for x in normalized_pantry if x]

        # merge pantry + user input
        query_ingredients = dedupe_preserve_order(
            user_ingredients + normalized_pantry
        )

        must_have_ingredients = [
            normalize_ingredient_text(row["value"])
            for row in st.session_state.ingredient_rows
            if row["value"].strip() and row["must_have"]
        ]

        must_have_ingredients = [x for x in must_have_ingredients if x]

        if not user_ingredients:
            st.warning("Please enter at least one typed ingredient. Pantry selections alone are not enough.")
            st.session_state.results = None
        elif not query_ingredients:
            st.warning("No valid ingredients were found after normalization.")
            st.session_state.results = None
        else:
            exclude_ingredients_list = [
                normalize_ingredient_text(x)
                for x in exclude_ingredients_text.split(",")
                if x.strip()
            ]

            exclude_ingredients_list = [x for x in exclude_ingredients_list if x]

            selected_include_cuisine = None if include_cuisine == "None" else include_cuisine
            selected_meal_type = None if meal_type == "None" else meal_type
            selected_dish_type = None if dish_type == "None" else dish_type
            selected_diet_label = None if diet_label == "None" else diet_label
            selected_min_match_count = None if min_match_count == 0 else int(min_match_count)

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
                    min_match_count=selected_min_match_count,
                    sort_by=sort_by,
                )

                # numeric cleanup
                results["calories"] = pd.to_numeric(results["calories"], errors="coerce")
                results["servings"] = pd.to_numeric(results["servings"], errors="coerce")

                # compute calories per serving
                results["calories_per_serving"] = results.apply(
                    lambda row: compute_calories_per_serving(
                        row.get("calories"),
                        row.get("servings"),
                    ),
                    axis=1,
                )

                # must-have filter
                if must_have_ingredients:
                    must_have_set = {x.lower() for x in must_have_ingredients}

                    results = results[
                        results["matched_ingredients"].apply(
                            lambda x: must_have_set.issubset(set(map(str.lower, x)))
                            if isinstance(x, list)
                            else False
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
                    errors="coerce",
                )
                missing_count = row.get("missing_count", 0)

                if pd.notna(calories_per_serving_value):
                    expander_title = (
                        f"{row['recipe_name']} • 🧂 {missing_count} missing • "
                        f"🍽️ {calories_per_serving_value:.0f} cal/serving"
                    )
                else:
                    expander_title = f"{row['recipe_name']} • 🧂 {missing_count} missing"

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

                        st.write(f"**Final Score:** {row.get('final_score', 0):.4f}")
                        st.write(f"**Similarity:** {row['similarity']:.4f}")
                        st.write(
                            f"**Matched Ingredients:** "
                            f"{render_list_value(row.get('matched_ingredients', []))}"
                        )
                        st.write(
                            f"**Extra Query Ingredients Not Used:** "
                            f"{render_list_value(row.get('extra_query_ingredients', []))}"
                        )
                        st.write(f"**Matched Count:** {row.get('matched_count', 0)}")
                        st.write(f"**Missing Count:** {missing_count}")
                        st.write(f"**Recipe Coverage:** {row.get('recipe_coverage', 0):.2%}")

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
                            st.write(
                                f"**Calories per serving:** "
                                f"{calories_per_serving_value:.0f}"
                            )
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
                        recipe.get("servings"),
                    )

                    if "final_score" in recipe:
                        st.write(f"**Final Score:** {float(recipe['final_score']):.4f}")

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
    st.write(
        "This recipe recommender takes user ingredients, converts them into the "
        "same vector space as recipe ingredients, retrieves similar recipes with "
        "cosine similarity, then reranks them using recommendation logic such as "
        "matched ingredients, recipe coverage, and missing ingredients."
    )