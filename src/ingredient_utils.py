import json
import re
from typing import Iterable, List, Optional


DESCRIPTOR_WORDS = {
    "fresh", "dried", "dry", "ground", "minced", "chopped", "sliced",
    "large", "small", "medium", "whole", "boneless", "skinless",
    "extra", "virgin", "extra-virgin", "lowfat", "low-fat",
    "reduced", "fat", "lean", "plain", "raw", "cooked"
}

INGREDIENT_ALIASES = {
    "extra virgin olive oil": "olive oil",
    "extra-virgin olive oil": "olive oil",
    "virgin olive oil": "olive oil",
    "evoo": "olive oil",

    "fresh thyme": "thyme",
    "dried thyme": "thyme",

    "fresh basil": "basil",
    "dried basil": "basil",

    "fresh parsley": "parsley",
    "italian parsley": "parsley",

    "kosher salt": "salt",
    "sea salt": "salt",

    "ground black pepper": "pepper",
    "black pepper": "pepper",

    "granulated sugar": "sugar",
    "white sugar": "sugar",

    "scallions": "green onion",
    "spring onions": "green onion",
}


def normalize_ingredient_text(text: str) -> Optional[str]:
    """
    Shared ingredient normalization used everywhere:
    - user query input
    - pantry items
    - dataset ingredient names
    - vectorization / similarity lookup
    """
    if not isinstance(text, str):
        return None

    text = text.lower().strip()

    # normalize separators
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")

    # remove punctuation / non-letters
    text = re.sub(r"[^a-z\s]", "", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    # direct alias match first
    if text in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[text]

    # remove descriptor words
    tokens = text.split()
    filtered_tokens = [tok for tok in tokens if tok not in DESCRIPTOR_WORDS]
    cleaned = " ".join(filtered_tokens).strip()

    if not cleaned:
        return None

    # alias match again after cleaning
    if cleaned in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[cleaned]

    return cleaned


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """
    Deduplicate strings while preserving order.
    Comparison is case-insensitive after stripping.
    """
    seen = set()
    result = []

    for item in items:
        if item is None:
            continue

        cleaned = str(item).strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def normalize_and_dedupe_ingredients(items: Iterable[str]) -> List[str]:
    """
    Normalize a sequence of ingredient strings and deduplicate them.
    """
    normalized = []

    for item in items:
        norm = normalize_ingredient_text(item)
        if norm:
            normalized.append(norm)

    return dedupe_preserve_order(normalized)


def safe_json_loads_list(x) -> List:
    """
    Safely parse a JSON list. Return [] if malformed or missing.
    """
    if isinstance(x, list):
        return x

    if x is None:
        return []

    try:
        # handles pandas NA / numpy nan safely
        if x != x:
            return []
    except Exception:
        pass

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