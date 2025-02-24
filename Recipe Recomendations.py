import pandas as pd
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = r"C:\Users\cmpas\Downloads\RAW_recipes.csv"
df = pd.read_csv(file_path)

# Reduce dataset size for efficiency (random sample of 500 recipes)
df_recipes = df[['name', 'description', 'ingredients', 'tags', 'nutrition', 'steps', 'minutes']].dropna().sample(n=500, random_state=42)

# Define keyword categories
CATEGORY_KEYWORDS = {
    "dessert": ["cookie", "cake", "brownie", "dessert", "sweets", "pie", "pastry", "ice cream", "pudding", "mousse", "tart", "sorbet", "cheesecake", "chocolate", "frosting"],
    "main_course": ["chicken", "beef", "pasta", "stew", "curry", "casserole", "dinner", "soup", "lasagna", "meatloaf", "bake", "stir fry", "grilled", "roasted", "tacos", "burrito", "sandwich", "wrap", "spaghetti", "sauteed", "stewed"],
    "healthy": ["healthy", "low calorie", "nutritious", "low fat", "high protein", "light", "low carb", "vegan", "vegetarian", "plant-based", "gluten-free", "organic", "whole grain", "fiber-rich", "sugar-free", "heart healthy"],
    "spicy": ["spicy", "hot", "chili", "cayenne", "pepper", "szechuan", "jalapeno"],
    "comfort_food": ["comfort food", "cheesy", "creamy", "fried", "hearty", "buttery", "rich"],
    "quick_meals": ["quick", "fast", "easy", "simple", "no-cook", "microwave", "15-minute"],
    "protein_rich": ["protein", "muscle", "gains", "high protein", "meat", "lean"],
    "breakfast": ["breakfast", "brunch", "pancake", "omelet", "smoothie", "cereal", "toast"],
    "seafood": ["fish", "salmon", "shrimp", "tuna", "lobster", "scallop", "crab", "oysters"]
}

# Function to clean text
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Combine text fields for searchability
df_recipes['combined_text'] = (
    df_recipes['description'].fillna('') + ' ' +
    df_recipes['ingredients'].astype(str).fillna('') + ' ' +
    df_recipes['tags'].astype(str).fillna('')
)

# Apply preprocessing
df_recipes['processed_text'] = df_recipes['combined_text'].apply(preprocess_text)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_recipes['processed_text'])

# Function to detect user preference category
def detect_category(user_input):
    user_input_lower = user_input.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return category
    return None

# Function to parse nutrition data
def parse_nutrition(nutrition_str):
    try:
        values = eval(nutrition_str)
        if len(values) >= 7:
            return {
                "Calories": values[0],
                "Total Fat": values[1],
                "Sugar": values[4],
                "Protein": values[5],
                "Sodium": values[6]
            }
    except:
        return None
    return None

df_recipes['nutrition_info'] = df_recipes['nutrition'].apply(parse_nutrition)

# Function to determine if a recipe is "healthy"
def is_healthy(nutrition_info):
    if not nutrition_info:
        return " Standard Meal"
    
    calories = nutrition_info.get("Calories", 1000)  
    fat = nutrition_info.get("Total Fat", 100)  
    sugar = nutrition_info.get("Sugar", 100)  
    protein = nutrition_info.get("Protein", 0)  
    sodium = nutrition_info.get("Sodium", 1000)  

    if calories < 500 and sugar < 10 and protein > 15 and fat < 20 and sodium < 600:
        return " Healthy Choice"
    return " Standard Meal"

# Apply health classification
df_recipes['health_rating'] = df_recipes['nutrition_info'].apply(is_healthy)

# Function to recommend recipes based on similarity and category
def recommend_recipes(user_input, top_n=5):
    category = detect_category(user_input)
    user_input_processed = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    top_indices = similarity_scores.argsort()[-top_n*2:][::-1]  # Get more than needed for filtering
    filtered_recommendations = []

    for index in top_indices:
        recipe = df_recipes.iloc[index]
        health_rating = recipe['health_rating']

        if category:
            # Ensure the recipe matches the detected category
            recipe_text = recipe['processed_text']
            category_keywords = CATEGORY_KEYWORDS[category]
            if not any(keyword in recipe_text for keyword in category_keywords):
                continue  # Skip if it does not match the desired category

        filtered_recommendations.append((recipe['name'], round(similarity_scores[index] * 100, 2), health_rating))

        if len(filtered_recommendations) >= top_n:
            break

    return filtered_recommendations

# Function to display recipe details
def show_recipe_details(recipe_name):
    recipe = df_recipes[df_recipes['name'] == recipe_name].iloc[0]

    print(f"\n **{recipe['name']}**")
    print(f" **Description:** {recipe['description']}")
    print(f" **Ingredients:** {recipe['ingredients']}")
    print(f" **Cooking Time:** {recipe['minutes']} minutes")
    print(f" **Health Classification:** {recipe['health_rating']}")  

    if recipe['nutrition_info']:
        print("\n **Nutritional Info:**")
        for key, value in recipe['nutrition_info'].items():
            print(f"   🔹 {key}: {value}")

    print("\n **Instructions:**")
    try:
        steps = eval(recipe['steps'])
        for step in steps:
            print(f"🔹 {step}")
    except:
        print(" Unable to display instructions.")

# Interactive user session
def interactive_session():
    while True:
        user_input = input("\n Enter a description of what you want to cook (or type 'exit' to quit): ").strip().lower()
        
        if user_input == "exit":
            print("\n👋 Goodbye! Happy Cooking! 🍳")
            break

        recommendations = recommend_recipes(user_input)

        if not recommendations:
            print("\n No matching recipes found. Try a different description.")
            continue

        print("\n **Top Recipe Recommendations:**")
        for i, (recipe, similarity, health_rating) in enumerate(recommendations, 1):
            print(f"{i}. {recipe} ({similarity}% match) - {health_rating}")

        while True:
            view_recipe = input("\n Enter a number to see details, 'new' for new recommendations, or 'exit': ").strip().lower()
            if view_recipe == "exit":
                return
            elif view_recipe == "new":
                break
            elif view_recipe.isdigit():
                selected_index = int(view_recipe) - 1
                if 0 <= selected_index < len(recommendations):
                    show_recipe_details(recommendations[selected_index][0])
                else:
                    print("\n Invalid selection.")

# Run the interactive session
if __name__ == "__main__":
    interactive_session()
