"""Weather retrieval and risk calc via OpenWeather API."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


def _get_risk(temperature: float, humidity: float) -> str:
    if temperature >= 30 and humidity >= 70:
        return "High"
    if temperature >= 25 and humidity >= 55:
        return "Medium"
    return "Low"


def get_weather(lat: float, lon: float) -> dict:
    if WEATHER_API_KEY is None or WEATHER_API_KEY.strip() == "":
        raise ValueError("WEATHER_API_KEY missing in environment")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    temp = float(data["main"]["temp"])
    humidity = float(data["main"]["humidity"])
    risk = _get_risk(temp, humidity)

    return {
        "temperature": temp,
        "humidity": humidity,
        "risk": risk,
    }
