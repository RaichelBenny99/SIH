"""Disease → treatment recommendations."""

def get_treatment(disease_name: str) -> dict:
    key = disease_name.replace("___", " ").replace("_", " ").strip().lower()

    database = {
        "potato early blight": {
            "organic": ["Remove and destroy infected leaves", "Use neem oil spray weekly"],
            "chemical": ["Apply fungicide with chlorothalonil", "Rotate with copper-based product"],
            "cultural": ["Ensure good air circulation", "Avoid overhead watering"],
        },
        "potato late blight": {
            "organic": ["Use compost tea", "Plant resistant varieties"],
            "chemical": ["Apply metalaxyl-m", "Use mancozeb"],
            "cultural": ["Destroy volunteer plants", "Maintain regular field sanitation"],
        },
        "tomato bacterial spot": {
            "organic": ["Copper sulfate spray", "Use disease-free seedling"],
            "chemical": ["Apply bactericide with copper", "Alternative: oxytetracycline"],
            "cultural": ["Avoid working while plants are wet", "Mulch the soil"],
        },
    }

    default = {
        "organic": ["Monitor leaf health frequently", "Remove severely infected tissues"],
        "chemical": ["Use broad-spectrum plant-safe treatment as per label", "Follow safety guidelines"],
        "cultural": ["Improve crop spacing", "Rotate crops annually"],
    }

    return database.get(key, default)
