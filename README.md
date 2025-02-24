# Recipe Recommendation System 

I’ve always loved cooking—experimenting with flavors, trying new recipes, and finding ways to make meals both delicious and nutritious. However one challenge I often faced was finding the perfect recipe when I had a specific craving or dietary preference in mind. Whether I wanted something spicy, healthy, or a quick meal, searching through countless recipes online can get annoying.

That’s what inspired me to build this Content-Based Recipe Recommendation System—a tool that helps people find the best recipes based on their own descriptions and preferences. By using TF-IDF vectorization and cosine similarity, the system can analyze a user’s input (e.g., "I want a high-protein spicy chicken dish") and return the most relevant recipes.

Beyond just matching recipes by text, I wanted this system to be smart about nutrition, so I integrated health classifications to help users identify whether a meal is a “Healthy Choice” or a “Standard Meal.”

This project combines my love for cooking with my passion for data science, and I hope it makes discovering new meals easier and more enjoyable for everyone!

---

## **Dataset**
- **Name**: [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- **Source**: Kaggle
- **File Used**: `RAW_recipes.csv`
- **Size**: 231,000+ recipes
- **Preprocessing**:
  - Unnecessary columns dropped
  - Missing values removed
  - Combined recipe descriptions, ingredients, and tags into a single text field
  - Nutrition data parsed for **calories, fat, sugar, protein, and sodium**
  - Sampled **500 recipes** for efficiency

---

## **Setup & Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/recipe-recommender.git
cd recipe-recommender
```
### **2. Create a Virtual Environment**

### For Windows
```
python -m venv venv
venv\Scripts\activate
````
### For Mac
```
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```
- pip install -r requirements.txt
```

### If pip is not recognized, use:
```
- python -m pip install -r requirements.txt
```

## **Download Dataset**

Go to the link in 'dataset' and download RAW_recipes.csv to your computer. Then copy and instert the file path into the 'file_path' function in the scr file.
```
file_path = r"C:\Users\cmpas\Downloads\RAW_recipes.csv"
df = pd.read_csv(file_path)
```

## **How the Code Works**
### **1. Text Vectorization**
The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors for comparison:

```
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_recipes['processed_text'])

```
The TF-IDF assigns higher weights to important words (e.g., "spicy", "vegan") and lower weights to common words** (e.g., "the", "and").

### **2. Recipe Category Detection**

The system automatically detects keywords in user input and prioritizes recipes that match relevant categories:
```
CATEGORY_KEYWORDS = {
    "dessert": ["cookie", "cake", "brownie", "dessert", "sweets"],
    "healthy": ["low calorie", "high protein", "vegan", "gluten-free"],
    "spicy": ["spicy", "hot", "chili", "jalapeno"],
}

def detect_category(user_input):
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            return category
    return None
```
If a user says "I want a spicy chicken dish", the system prioritizes the words'spicy' and 'chicken' recipes.


### **3. Finding the Best Match**

Using cosine similarity, the system compares the user’s request to all recipes and finds the most similar ones:

```
user_vector = vectorizer.transform([user_input])
similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
top_indices = similarity_scores.argsort()[-5:][::-1]  # Sort in descending order

```
Higher similarity scores = Better recipe matches
The system returns the top 5 most relevant recipes.

### **4. Determining If a Recipe Is Healthy**

Each recipe contains nutrition data (Calories, Fat, Sugar, Protein, Sodium). The system classifies meals based on the following criteria:

Criteria to be a	Healthy Threshold

Calories	< 500 kcal
Total Fat	< 20g
Sugar	< 10g
Protein	> 15g
Sodium	< 600mg

If a recipe meets these requirements, it is labeled "Healthy Choice":

```
def is_healthy(nutrition_info):
    if not nutrition_info:
        return "Standard Meal"
    
    if nutrition_info["Calories"] < 500 and nutrition_info["Sugar"] < 10 and nutrition_info["Protein"] > 15:
        return "Healthy Choice"
    return "Standard Meal"
```

## **Example Output**

### **User Input:**

```
Enter a description of what you want to cook (or type 'exit' to quit): low fat pasta dish
```

### **System Output:**

```
 **Top Recipe Recommendations:**
1. Low Fat Italian Pasta Salad (41.34% match) - Standard Meal
2. Spicy Sweet Mustard Chicken (28.45% match) - Standard Meal
3. Honey Glazed Shallots with Mint (18.41% match) - Standard Meal
4. Sausage and Roasted Peppers Pasta Bake (17.41% match) - Standard Meal
5. Curry Chicken Pasta (15.82% match) - Standard Meal

Enter a number to see details, 'new' for new recommendations, or 'exit': 1
```

### **User selects option 1 (Low Fat Italian Pasta Salad):**

```
 **Low Fat Italian Pasta Salad**
 **Description:** Low fat, low cholesterol Italian-flavored pasta salad.  
 I'm now on a low-fat, low-cholesterol diet. I don't eat a lot of "good-for-you" foods. The ones I do eat,
 I have to eat them cooked—don't like crunchy foods.  
 Often, I just throw a bunch of stuff together and see how I like it. This is how I got this one.    

 **Ingredients:**  
- Fat-free Italian salad dressing  
- Water  
- Zucchini  
- Roma tomato  
- Onion  
- Garlic  
- Bell pepper  
- Rotini pasta  
- Basil  
- Marjoram  
- Rosemary  
- Garlic seasoning  
- Croutons  

 **Cooking Time:** 25 minutes  
 **Health Classification:** Standard Meal  

 **Nutritional Info:**  
   🔹 **Calories:** 21.6  
   🔹 **Total Fat:** 0.0g  
   🔹 **Sugar:** 2.0g  
   🔹 **Protein:** 0.0g  
   🔹 **Sodium:** 1.0mg  

 **Instructions:**  
🔹 Cook pasta.  
🔹 Meanwhile, sauté garlic, onions, bell pepper, zucchini, and Roma tomato together in the fat-free Italian dressing and a cup of water, adding spices (basil, marjoram, and rosemary).  
🔹 Then, add the pasta and top with salad toppers, low-fat parmesan, red pepper, and garlic seasoning.  

Enter a number to see details, 'new' for new recommendations, or 'exit':
```

## Salary expectation per month

I would expect somewhere between 1600-2400$ a month, given 20/30$ an hour for 20 hours a week.

