import pandas as pd
from ast import literal_eval
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import google.auth
from google import genai
from google.genai import types
from tqdm import tqdm
from sklearn.model_selection import train_test_split

load_dotenv()
CRED, PROJECT_ID = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
LOCATION = "europe-west4"
SEARCH_TOOL = TavilySearchResults(max_results=5, output_format="list")
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", credentials=CRED, temperature=0)
ALLOWED_CATEGORIES = [
        "vegetables",
        "fruits",
        "dairy products",
        "cereals & starchy foods",
        "condiments & spices",
        "nuts",
        "meats & fishes",
    ]

class NutritionPer100g(BaseModel):
    kcal: Optional[float] = Field(None, description="Energy in kcal per 100g")
    protein_g: Optional[float] = Field(None, description="Protein (g) per 100g")
    carbs_g: Optional[float] = Field(None, description="Carbohydrates (g) per 100g")
    fat_g: Optional[float] = Field(None, description="Total fat (g) per 100g")
    fiber_g: Optional[float] = Field(None, description="Fiber (g) per 100g")
    sugar_g: Optional[float] = Field(None, description="Total sugars (g) per 100g")
    sodium_mg: Optional[float] = Field(None, description="Sodium (mg) per 100g")


class IngredientResult(BaseModel):
    ingredient: str = Field(..., description="Ingredient name")
    category: str = Field(..., description=f"One of: {ALLOWED_CATEGORIES}")
    nutrition_per_100g: NutritionPer100g = Field(
        ...,
        description="Nutrition values per 100g"
    )
    sources: list[str] = Field(..., description="URLs or source names used")


class IngredientState(TypedDict):
    ingredient: str
    search_results: str
    result: Optional[dict[str, Any]]

def web_search_node(state: IngredientState) -> IngredientState:
    """Run a web search for an ingredient and store serialized results in state.

    Args:
        state: Current ingredient processing state.

    Returns:
        Updated state containing JSON-formatted search results.
    """
    q = f"{state['ingredient']} nutrition per 100g USDA category food group"
    hits = SEARCH_TOOL.invoke(q)
    return {
        **state,
        "search_results": json.dumps(hits, ensure_ascii=False, indent=2)
    }

def extract_node(state: IngredientState) -> IngredientState:
    """Extract category and nutrition per 100g from search results using the LLM.

    Args:
        state: Current ingredient processing state with search snippets.

    Returns:
        Updated state containing a parsed structured result.
    """
    prompt = f"""
You are given an ingredient and web search snippets.
Task:
1) Infer the ingredient category from this exact list only:
{ALLOWED_CATEGORIES}
2) Extract nutrition data per 100g from reliable sources (prefer USDA/FDC-like sources).
3) Return JSON only.

Ingredient: {state['ingredient']}

Web search snippets:
{state['search_results']}
""" 
    structured_llm = LLM.with_structured_output(IngredientResult)
    parsed = structured_llm.invoke(prompt)
    return {**state, "result": parsed.model_dump()}

def build_ingredient_graph() -> Any:
    """Build and compile the ingredient enrichment state graph.

    Returns:
        Compiled LangGraph workflow used to process ingredient states.
    """
    # ---- Graph ----
    builder = StateGraph(IngredientState)
    builder.add_node("web_search", web_search_node)
    builder.add_node("extract", extract_node)

    builder.set_entry_point("web_search")
    builder.add_edge("web_search", "extract")
    builder.add_edge("extract", END)

    ingredient_graph = builder.compile()
    return ingredient_graph

def group_ingredients(ingredient_df: pd.DataFrame) -> dict[str, str]:
    """Group ingredient name variants under a shared canonical name using an LLM.

    Args:
        ingredient_df: DataFrame containing ingredient profiles with a 'canonical' column.

    Returns:
        Dictionary mapping each ingredient variant (lowercased, stripped) to its
        canonical ingredient name.
    """
    genai_client = client = genai.Client(
    vertexai=True, credentials=CRED, project=PROJECT_ID, location=LOCATION
)

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "canonical_name": {"type": "STRING"},
                    "variants": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": ["canonical_name", "variants"],
            },
        },
    )


    category_grouping_prompts = {}
    category_groupings = {}

    for category, group in ingredient_df.groupby("category"):
        ingredient_names = sorted(group["ingredient"].dropna().astype(str).str.strip().unique().tolist())
        ingredient_list = "\n".join(f"- {name}" for name in ingredient_names)

        prompt = f"""
You are a food taxonomy assistant.
Category: {category}

Task:
Group ingredient names that refer to the same product (spelling variants, plural/singular, close synonyms).
Return valid ARRAY only in this format:
[
{{
    "canonical_name": "string",
    "variants": ["string", "string"]
}}
]

Rules:
- Keep each ingredient in exactly one group.
- Prefer the most common/simple canonical food name.
- Do not invent new ingredients outside the list.

Ingredient names:
{ingredient_list}
    """.strip()

        category_grouping_prompts[category] = prompt

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=generation_config,
        ).text

        try:
            category_groupings[category] = json.loads(response)
        except Exception:
            category_groupings[category] = {"category": category, "raw_response": response}
    variant_to_canonical = {}
    for cat, groupings in category_groupings.items():
        if isinstance(groupings, list):
            for group in groupings:
                canonical = group["canonical_name"]
                for variant in group["variants"]:
                    variant_to_canonical[variant.strip().lower()] = canonical
        elif isinstance(groupings, dict) and "raw_response" in groupings:
            try:
                parsed = json.loads(groupings["raw_response"])
                for group in parsed:
                    canonical = group["canonical_name"]
                    for variant in group["variants"]:
                        variant_to_canonical[variant.strip().lower()] = canonical
            except Exception as e:
                print(f"Could not parse raw_response for category '{cat}': {e}")
    return variant_to_canonical


def get_preferred_value(
    group_df: pd.DataFrame, col: str, canonical_name: str
) -> Any:
    """Return the best available value for a column within a group of ingredient rows.

    Preference is given to the row whose ingredient name matches the canonical name
    exactly. Falls back to the first non-null value across the group.

    Args:
        group_df: Subset of the ingredient DataFrame sharing the same canonical name.
        col: Column name whose value should be retrieved.
        canonical_name: Canonical ingredient name used for exact-match lookup.

    Returns:
        The preferred non-null value for the column, or ``None`` if no value is found.
    """
    exact_match = group_df[
        group_df["ingredient"].str.strip().str.lower() == canonical_name
    ][col].dropna()
    if not exact_match.empty:
        return exact_match.iloc[0]

    non_null = group_df[col].dropna()
    return non_null.iloc[0] if not non_null.empty else None

def merge_group(
    group_df: pd.DataFrame, canonical_name: str, numeric_cols: list[str]
) -> pd.Series:
    """Merge a group of ingredient variant rows into a single representative row.

    For each numeric column the best available value is selected via
    :func:`get_preferred_value`, preferring the canonical ingredient's own row.

    Args:
        group_df: Subset of the ingredient DataFrame sharing the same canonical name.
        canonical_name: Canonical ingredient name assigned to the merged row.
        numeric_cols: List of numeric column names to aggregate.

    Returns:
        A :class:`pandas.Series` with ``ingredient``, ``category``, and one entry
        per column in *numeric_cols*.
    """
    # For each numeric column, take the first non-null value from canonical or variants
    result = {"ingredient": canonical_name, "category": group_df["category"].iloc[0]}
    for col in numeric_cols:
        result[col] = get_preferred_value(group_df, col, canonical_name)
    return pd.Series(result)
    


if __name__ == "__main__":

    recipes = pd.read_csv('gs://recipe-generation/recipes.csv', sep=";")
    recipes = recipes.dropna(subset=['Ingredients'])
    ingredients_count = recipes['Ingredients'].apply(lambda l : [val["ingredient"] for val in literal_eval(l)]).explode().value_counts()
    top_ingredients = ingredients_count.head(500)
    filtered_recipes = recipes[recipes['Ingredients'].apply(
        lambda l: all(val["ingredient"] in top_ingredients.index for val in literal_eval(l))
    )]

    ingredient_data = []
    ingredient_graph = build_ingredient_graph()
    for ingredient in tqdm(top_ingredients.index, desc="Processing ingredients"):
        print(f"Processing ingredient: {ingredient}")
        initial_state = IngredientState(ingredient=ingredient, search_results="", result=None)
        final_state = ingredient_graph.invoke(initial_state)

        # Check if all nutrition values are None
        nutrition_keys = ["kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
        nutrition_values = [
            final_state['result']['nutrition_per_100g'][key] if final_state['result'] else None
            for key in nutrition_keys
        ]
        if all(val is None for val in nutrition_values):
            # Re-run the graph for this ingredient (up to 2 attempts)
            print(f"All nutrition values missing for '{ingredient}', retrying...")
            final_state = ingredient_graph.invoke(initial_state)
            nutrition_values = [
                final_state['result']['nutrition_per_100g'][key] if final_state['result'] else None
                for key in nutrition_keys
            ]
        # Replace None with 0 if only some values are missing
        nutrition_values = [0 if val is None else val for val in nutrition_values]

        ingredient_data.append({
            "ingredient": ingredient,
            "category": final_state['result']['category'] if final_state['result'] else None,
            "kcal": nutrition_values[0],
            "protein_g": nutrition_values[1],
            "carbs_g": nutrition_values[2],
            "fat_g": nutrition_values[3],
            "fiber_g": nutrition_values[4],
            "sugar_g": nutrition_values[5],
            "sodium_mg": nutrition_values[6],
        })
    ingredient_df = pd.DataFrame(ingredient_data)
    variant_to_canonical = group_ingredients(ingredient_df)

    filtered_recipes['Ingredients'] = filtered_recipes['Ingredients'].apply(lambda l : [variant_to_canonical.get(val["ingredient"].strip().lower(), val["ingredient"].strip().lower()) for val in literal_eval(l)])

    train, test = train_test_split(filtered_recipes, test_size=0.2, random_state=42)
    train.to_csv('gs://recipe-generation/train_recipes.csv', index=False)
    test.to_csv('gs://recipe-generation/test_recipes.csv', index=False)

    ingredient_df["canonical"] = ingredient_df["ingredient"].str.strip().str.lower().map(variant_to_canonical)
    ingredient_df["canonical"] = ingredient_df["canonical"].fillna(ingredient_df["ingredient"].str.strip().str.lower())

    ingredient_df_normalized = pd.DataFrame(columns=["ingredient", "category"] + nutrition_keys)
    for canonical_name, group in ingredient_df.groupby("canonical", sort=False):
        merged = merge_group(group, canonical_name, nutrition_keys)
        ingredient_df_normalized = pd.concat([ingredient_df_normalized, merged.to_frame().T], ignore_index=True)

    print(f"ingredient_df: {len(ingredient_df)} rows -> {len(ingredient_df_normalized)} rows after normalization")

    ingredient_df_normalized.to_csv('gs://recipe-generation/ingredient_profiles.csv', index=False)
