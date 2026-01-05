"""
Weather Service Module

Provides real-time weather data from Open-Meteo API for agriculture applications.
Features:
- Current weather conditions
- 7-day forecasts
- Agricultural metrics (evapotranspiration, soil data)
- Geocoding (city name → coordinates)
- LRU caching with configurable TTL
"""
import requests
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from functools import lru_cache
import time

from ..config.config import logger


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WeatherData:
    """Current weather conditions."""
    location: str
    latitude: float
    longitude: float
    temperature: float  # Celsius
    humidity: float  # Percentage
    precipitation: float  # mm
    wind_speed: float  # km/h
    description: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "precipitation": self.precipitation,
            "wind_speed": self.wind_speed,
            "description": self.description,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DailyForecast:
    """Daily weather forecast."""
    date: str
    temp_min: float
    temp_max: float
    precipitation_sum: float
    precipitation_probability: float
    wind_speed_max: float
    weather_code: int
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "temp_min": self.temp_min,
            "temp_max": self.temp_max,
            "precipitation_sum": self.precipitation_sum,
            "precipitation_probability": self.precipitation_probability,
            "wind_speed_max": self.wind_speed_max,
            "weather_code": self.weather_code,
            "description": self.description
        }


@dataclass
class AgriculturalWeatherData:
    """Weather data with agricultural metrics."""
    location: str
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    evapotranspiration: float  # mm/day (ET₀)
    soil_temperature: Optional[float] = None  # 0-10cm depth
    soil_moisture: Optional[float] = None  # 0-10cm depth
    forecast: List[DailyForecast] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "precipitation": self.precipitation,
            "wind_speed": self.wind_speed,
            "evapotranspiration": self.evapotranspiration,
            "soil_temperature": self.soil_temperature,
            "soil_moisture": self.soil_moisture,
            "forecast": [f.to_dict() for f in self.forecast],
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_irrigation_recommendation(self) -> str:
        """Generate basic irrigation recommendation based on weather."""
        if self.precipitation > 5:
            return "No irrigation needed - recent/expected rainfall"
        elif self.evapotranspiration > 6:
            return "High evapotranspiration - increase irrigation"
        elif self.temperature > 35:
            return "High temperature - consider early morning irrigation"
        elif self.humidity < 30:
            return "Low humidity - monitor soil moisture closely"
        else:
            return "Normal conditions - follow standard irrigation schedule"


# =============================================================================
# Weather Code Descriptions
# =============================================================================

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather_description(code: int) -> str:
    """Convert WMO weather code to description."""
    return WMO_WEATHER_CODES.get(code, "Unknown")


# =============================================================================
# Weather Service
# =============================================================================

class WeatherService:
    """
    Weather service client for Open-Meteo API.
    
    Provides:
    - Current weather conditions
    - 7-day forecasts
    - Agricultural metrics (ET₀, soil data)
    - City name geocoding
    
    Features:
    - LRU caching with TTL
    - Retry logic for API failures
    - Timeout handling
    """
    
    # Open-Meteo API endpoints
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(
        self,
        cache_ttl: int = 1800,  # 30 minutes
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize the weather service.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            timeout: API request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Cache storage with timestamps
        self._cache: Dict[str, Tuple[Any, float]] = {}
        
        logger.info("WeatherService initialized with Open-Meteo API")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for: {key}")
                return value
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any):
        """Store value in cache with timestamp."""
        self._cache[key] = (value, time.time())
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(1)  # Brief delay before retry
        
        return None
    
    def geocode_location(self, location: str) -> Optional[Tuple[float, float, str]]:
        """
        Convert location name to coordinates.
        
        Args:
            location: City, village, or region name
            
        Returns:
            Tuple of (latitude, longitude, full_name) or None if not found
        """
        cache_key = f"geo:{location.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        data = self._make_request(self.GEOCODING_URL, params)
        
        if data and "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            coords = (
                result["latitude"],
                result["longitude"],
                f"{result.get('name', location)}, {result.get('country', '')}"
            )
            self._set_cache(cache_key, coords)
            logger.info(f"Geocoded '{location}' to {coords[2]} ({coords[0]}, {coords[1]})")
            return coords
        
        logger.warning(f"Could not geocode location: {location}")
        return None
    
    def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """
        Get current weather conditions for a location.
        
        Args:
            location: City, village, or region name
            
        Returns:
            WeatherData object or None if failed
        """
        coords = self.geocode_location(location)
        if not coords:
            return None
        
        lat, lon, full_name = coords
        cache_key = f"current:{lat},{lon}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "auto"
        }
        
        data = self._make_request(self.WEATHER_URL, params)
        
        if data and "current" in data:
            current = data["current"]
            weather = WeatherData(
                location=full_name,
                latitude=lat,
                longitude=lon,
                temperature=current.get("temperature_2m", 0),
                humidity=current.get("relative_humidity_2m", 0),
                precipitation=current.get("precipitation", 0),
                wind_speed=current.get("wind_speed_10m", 0),
                description=get_weather_description(current.get("weather_code", 0)),
                timestamp=datetime.now()
            )
            self._set_cache(cache_key, weather)
            return weather
        
        return None
    
    def get_forecast(
        self, 
        location: str, 
        days: int = 7
    ) -> Optional[List[DailyForecast]]:
        """
        Get weather forecast for a location.
        
        Args:
            location: City, village, or region name
            days: Number of forecast days (1-16)
            
        Returns:
            List of DailyForecast objects or None if failed
        """
        coords = self.geocode_location(location)
        if not coords:
            return None
        
        lat, lon, _ = coords
        cache_key = f"forecast:{lat},{lon}:{days}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,weather_code",
            "timezone": "auto",
            "forecast_days": min(days, 16)
        }
        
        data = self._make_request(self.WEATHER_URL, params)
        
        if data and "daily" in data:
            daily = data["daily"]
            forecasts = []
            
            for i in range(len(daily.get("time", []))):
                forecast = DailyForecast(
                    date=daily["time"][i],
                    temp_min=daily.get("temperature_2m_min", [0])[i],
                    temp_max=daily.get("temperature_2m_max", [0])[i],
                    precipitation_sum=daily.get("precipitation_sum", [0])[i],
                    precipitation_probability=daily.get("precipitation_probability_max", [0])[i],
                    wind_speed_max=daily.get("wind_speed_10m_max", [0])[i],
                    weather_code=daily.get("weather_code", [0])[i],
                    description=get_weather_description(daily.get("weather_code", [0])[i])
                )
                forecasts.append(forecast)
            
            self._set_cache(cache_key, forecasts)
            return forecasts
        
        return None
    
    def get_agricultural_metrics(
        self, 
        location: str,
        forecast_days: int = 7
    ) -> Optional[AgriculturalWeatherData]:
        """
        Get comprehensive agricultural weather data.
        
        Includes:
        - Current conditions
        - ET₀ (reference evapotranspiration)
        - Soil temperature and moisture (when available)
        - Multi-day forecast
        
        Args:
            location: City, village, or region name
            forecast_days: Number of forecast days
            
        Returns:
            AgriculturalWeatherData object or None if failed
        """
        coords = self.geocode_location(location)
        if not coords:
            return None
        
        lat, lon, full_name = coords
        cache_key = f"agri:{lat},{lon}:{forecast_days}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Request comprehensive data
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,weather_code,et0_fao_evapotranspiration",
            "hourly": "soil_temperature_0cm,soil_moisture_0_to_1cm",
            "timezone": "auto",
            "forecast_days": min(forecast_days, 16)
        }
        
        data = self._make_request(self.WEATHER_URL, params)
        
        if not data:
            return None
        
        # Parse current conditions
        current = data.get("current", {})
        
        # Parse daily forecast
        daily = data.get("daily", {})
        forecasts = []
        et0_values = daily.get("et0_fao_evapotranspiration", [0])
        
        for i in range(len(daily.get("time", []))):
            forecast = DailyForecast(
                date=daily["time"][i],
                temp_min=daily.get("temperature_2m_min", [0])[i],
                temp_max=daily.get("temperature_2m_max", [0])[i],
                precipitation_sum=daily.get("precipitation_sum", [0])[i],
                precipitation_probability=daily.get("precipitation_probability_max", [0])[i],
                wind_speed_max=daily.get("wind_speed_10m_max", [0])[i],
                weather_code=daily.get("weather_code", [0])[i],
                description=get_weather_description(daily.get("weather_code", [0])[i])
            )
            forecasts.append(forecast)
        
        # Parse hourly soil data (get most recent values)
        hourly = data.get("hourly", {})
        soil_temp = None
        soil_moisture = None
        
        if hourly.get("soil_temperature_0cm"):
            # Get the most recent non-null value
            soil_temps = [t for t in hourly["soil_temperature_0cm"] if t is not None]
            soil_temp = soil_temps[-1] if soil_temps else None
        
        if hourly.get("soil_moisture_0_to_1cm"):
            soil_moistures = [m for m in hourly["soil_moisture_0_to_1cm"] if m is not None]
            soil_moisture = soil_moistures[-1] if soil_moistures else None
        
        # Calculate today's ET₀ or average
        today_et0 = et0_values[0] if et0_values else 0
        
        agri_data = AgriculturalWeatherData(
            location=full_name,
            latitude=lat,
            longitude=lon,
            temperature=current.get("temperature_2m", 0),
            humidity=current.get("relative_humidity_2m", 0),
            precipitation=current.get("precipitation", 0),
            wind_speed=current.get("wind_speed_10m", 0),
            evapotranspiration=today_et0,
            soil_temperature=soil_temp,
            soil_moisture=soil_moisture,
            forecast=forecasts,
            timestamp=datetime.now()
        )
        
        self._set_cache(cache_key, agri_data)
        logger.info(f"Retrieved agricultural weather data for {full_name}")
        
        return agri_data
    
    def format_for_llm(
        self, 
        data: AgriculturalWeatherData,
        include_forecast: bool = True
    ) -> str:
        """
        Format weather data as context for LLM.
        
        Args:
            data: Agricultural weather data
            include_forecast: Whether to include forecast details
            
        Returns:
            Formatted string for LLM context
        """
        lines = [
            f"=== WEATHER CONTEXT: {data.location} ===",
            f"Retrieved: {data.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            "CURRENT CONDITIONS:",
            f"  • Temperature: {data.temperature:.1f}°C",
            f"  • Humidity: {data.humidity:.0f}%",
            f"  • Precipitation: {data.precipitation:.1f} mm",
            f"  • Wind Speed: {data.wind_speed:.1f} km/h",
            "",
            "AGRICULTURAL METRICS:",
            f"  • Reference Evapotranspiration (ET₀): {data.evapotranspiration:.2f} mm/day",
        ]
        
        if data.soil_temperature is not None:
            lines.append(f"  • Soil Temperature (0-10cm): {data.soil_temperature:.1f}°C")
        
        if data.soil_moisture is not None:
            lines.append(f"  • Soil Moisture (0-10cm): {data.soil_moisture:.3f} m³/m³")
        
        lines.extend([
            "",
            f"IRRIGATION RECOMMENDATION: {data.get_irrigation_recommendation()}",
        ])
        
        if include_forecast and data.forecast:
            lines.extend(["", "7-DAY FORECAST:"])
            for day in data.forecast[:7]:
                rain_icon = "🌧️" if day.precipitation_probability > 50 else "☀️"
                lines.append(
                    f"  {day.date}: {day.temp_min:.0f}-{day.temp_max:.0f}°C, "
                    f"{rain_icon} {day.precipitation_probability:.0f}% rain, "
                    f"{day.description}"
                )
        
        return "\n".join(lines)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Weather cache cleared")


# =============================================================================
# Module-level Instance
# =============================================================================

# Lazy-loaded singleton
_weather_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    """Get or create the weather service singleton."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


# =============================================================================
# Convenience Functions
# =============================================================================

def get_weather_for_location(location: str) -> Optional[AgriculturalWeatherData]:
    """Convenience function to get weather data."""
    return get_weather_service().get_agricultural_metrics(location)


def format_weather_context(location: str) -> Optional[str]:
    """Convenience function to get formatted weather context for LLM."""
    service = get_weather_service()
    data = service.get_agricultural_metrics(location)
    if data:
        return service.format_for_llm(data)
    return None


# =============================================================================
# Main (Testing)
# =============================================================================

def main():
    """Test the weather service."""
    print("Weather Service Test")
    print("=" * 60)
    
    service = WeatherService()
    
    # Test geocoding
    test_locations = ["Marrakech", "Casablanca", "Fes", "Agadir"]
    
    for location in test_locations:
        print(f"\nTesting: {location}")
        print("-" * 40)
        
        # Get agricultural data
        data = service.get_agricultural_metrics(location)
        
        if data:
            print(service.format_for_llm(data, include_forecast=True))
        else:
            print(f"Failed to get weather data for {location}")


if __name__ == "__main__":
    main()
