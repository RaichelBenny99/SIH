"""Gemini LLM helper for agronomist conversation."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def ask_agronomist(disease: str, severity: dict, weather: dict, user_question: str) -> dict:
    if GEMINI_API_KEY is None or GEMINI_API_KEY.strip() == "":
        raise ValueError("GEMINI_API_KEY missing in environment")

    prompt = (
        f"You are an expert agronomist. A farmer has a plant disease identified as '{disease}' "
        f"with severity {severity.get('severity', severity.get('severity_label', 'Unknown'))} "
        f"({severity.get('infected_pct', severity.get('severity_score', 0))}%). "
        f"Current weather is {weather.get('temperature')}°C, humidity {weather.get('humidity')}%, "
        f"risk level {weather.get('risk')}. "
        f"Answer the question concisely with practical steps: {user_question}"
    )

    def local_fallback():
        disease_fallback = {
            'tomato septoria leaf spot': 'Remove infected leaves, apply copper fungicide, and avoid overhead irrigation.',
            'potato early blight': 'Improve airflow, remove infected foliage, apply mancozeb or chlorothalonil, and rotate crops.',
            'potato late blight': 'Use resistant varieties, apply metalaxyl-m or mancozeb, avoid wet conditions.',
        }
        key = disease.lower().replace('_', ' ').replace('___', ' ').strip()
        advice = disease_fallback.get(key, 'Maintain crop hygiene, use resistant varieties, and apply approved fungicides per label.')
        return f"**Offline recommendation:** {advice}"

    text = None
    error_msg = None
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    if HAS_GENAI:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1500,
                    top_p=0.95,
                ),
            )
            text = response.text if hasattr(response, 'text') else None
            if text:
                error_msg = None
        except Exception as e:
            error_msg = f"Gemini SDK ({model_name}) failed: {str(e)[:100]}"

    if text is None or text.strip() == "":
        # Final fallback: use local heuristic advice
        text = local_fallback()

    return {
        "query": user_question,
        "response": text.strip() if text else "",
        "model": model_name,
        "error": error_msg,
    }
