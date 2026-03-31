# 🍽️ Recipe Recommendation System

A Streamlit-based recipe recommender that suggests recipes based on user-provided ingredients, with support for filtering, bookmarks, and interactive exploration.

---

## 🚀 Features

### 🔍 Ingredient-Based Search
- Enter ingredients dynamically (one at a time)
- Add multiple ingredients using a "+" button
- Mark ingredients as **must-have** for strict filtering

### 🎯 Smart Recommendation System
- Uses similarity-based retrieval to rank recipes
- Returns top matching recipes based on input ingredients
- Expands candidate pool before filtering for better results

### ⚙️ Advanced Filters
- Cuisine (include / exclude)
- Meal type
- Dish type
- Diet labels
- Servings (min / max)
- Calories **per serving** (min / max)
- Pantry-only mode (only recipes you can fully make)

### 📊 Informative Results
Each recipe shows:
- Missing ingredient count 🧂
- Calories per serving 🍽️
- Total calories
- Servings
- Matched vs missing ingredients
- Full ingredient list (expandable)
- Direct link to recipe

### 📌 Bookmarks
- Save recipes for later
- View all saved recipes on a dedicated page
- Remove bookmarks anytime

### 🧭 Multi-Page App
Sidebar navigation includes:
- Recipe Search
- Bookmarks
- PCA Visualization (placeholder)
- About

---

## 🖥️ Demo UI Highlights

- Expandable result cards for clean browsing
- Emoji-enhanced summaries for quick comparison
- Compact filter panel
- Side-by-side layout for images and details

---

## 🧠 How It Works

1. User inputs ingredients
2. Ingredients are normalized and processed
3. Recipes are retrieved using similarity scoring
4. A larger candidate pool is selected
5. Filters are applied:
   - must-have ingredients
   - dietary constraints
   - calories per serving
   - servings
6. Top results are displayed interactively

---
## 📂 Project Structure

```
recipe-recommender/
│
├── app.py                  # Streamlit app
├── src/
│   ├── similarity.py       # Retrieval + ranking logic
│   ├── filtering.py        # Filtering utilities
│   └── data_processing.py  # Data cleaning + normalization
│
├── models/
│   └── recipe_metadata.csv # Cleaned recipe dataset
│
└── README.md
```
---

## 🖥️ Working Environment

The application was developed and tested with the following environment:

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Python | 3.12.3 |
| Streamlit | 1.55.0 |
| Pandas | 2.3.3 |
| NumPy | 2.4.4 |
| scikit-learn | 1.8.0 |

> **Note:** Other versions may work, but the above configuration is confirmed to run correctly.

---

## 🛠️ Installation

### 1. Clone the repo
```
git clone https://github.com/your-username/recipe-recommender.git
cd recipe-recommender
```
### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the app
```
streamlit run app.py
```

---

## 📌 Example Usage

- Input: `chicken, garlic`
- Mark `chicken` as must-have
- Set max calories per serving
- View recipes ranked by similarity
- Save favorites to bookmarks

---

## 🔮 Future Improvements

- PCA visualization of recipe embeddings
- Clustering recipes by cuisine/type
- Shopping list generator (missing ingredients)
- Sorting (by calories, similarity, missing count)
- User preference learning
