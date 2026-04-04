import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key found: {api_key[:20]}...")

# Test available models
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    print("\n=== Testing SDK API ===")
    
    # List available models
    print("Available models:")
    for model in genai.list_models():
        print(f"  - {model.name}")
    
    # Try to generate with gemini-pro
    print("\n=== Testing gemini-1.5-flash ===")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Say hello')
        print(f"✓ gemini-1.5-flash works: {response.text[:50]}")
    except Exception as e:
        print(f"✗ gemini-1.5-flash failed: {e}")
    
    # Try gemini-pro
    print("\n=== Testing gemini-pro ===")
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content('Say hello')
        print(f"✓ gemini-pro works: {response.text[:50]}")
    except Exception as e:
        print(f"✗ gemini-pro failed: {e}")
        
except Exception as e:
    print(f"SDK import/config error: {e}")
