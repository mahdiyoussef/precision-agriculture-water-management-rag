"""
Infrastructure Package

External adapters for:
- llm: LLM service integrations
- persistence: Storage adapters
- weather_service: Weather data from Open-Meteo API
"""

from .weather_service import (
    WeatherService,
    WeatherData,
    AgriculturalWeatherData,
    DailyForecast,
    get_weather_service,
    get_weather_for_location,
    format_weather_context
)

__all__ = [
    "WeatherService",
    "WeatherData", 
    "AgriculturalWeatherData",
    "DailyForecast",
    "get_weather_service",
    "get_weather_for_location",
    "format_weather_context"
]
