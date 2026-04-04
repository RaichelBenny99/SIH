import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Test the latest models
for model_name in ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-flash-latest']:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content('What should I do for tomato septoria leaf spot?', generation_config={'max_output_tokens': 100, 'temperature': 0.2})
        print(f"✓ {model_name} WORKS:")
        print(f"  Response: {response.text[:100]}...")
        break
    except Exception as e:
        print(f"✗ {model_name}: {e}")
