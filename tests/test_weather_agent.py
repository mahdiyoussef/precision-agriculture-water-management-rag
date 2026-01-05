"""
Unit Tests for Weather Agent and Weather Service

Tests for:
- Weather relevance detection
- Location geocoding
- API response parsing
- Weather context formatting
- Caching behavior
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Weather Service Tests
# ============================================================================

class TestWeatherData:
    """Test WeatherData and related dataclasses."""
    
    def test_weather_data_creation(self):
        """Test WeatherData object creation."""
        from src.infrastructure.weather_service import WeatherData
        
        data = WeatherData(
            location="Marrakech, Morocco",
            latitude=31.63,
            longitude=-8.0,
            temperature=25.0,
            humidity=45.0,
            precipitation=0.0,
            wind_speed=12.5,
            description="Clear sky",
            timestamp=datetime.now()
        )
        
        assert data.location == "Marrakech, Morocco"
        assert data.temperature == 25.0
        assert data.humidity == 45.0
    
    def test_weather_data_to_dict(self):
        """Test dictionary conversion."""
        from src.infrastructure.weather_service import WeatherData
        
        data = WeatherData(
            location="Test",
            latitude=0.0,
            longitude=0.0,
            temperature=20.0,
            humidity=50.0,
            precipitation=0.0,
            wind_speed=10.0,
            description="Test",
            timestamp=datetime.now()
        )
        
        d = data.to_dict()
        assert isinstance(d, dict)
        assert d["temperature"] == 20.0
        assert "timestamp" in d


class TestAgriculturalWeatherData:
    """Test AgriculturalWeatherData with irrigation recommendations."""
    
    def test_irrigation_recommendation_rain(self):
        """Test recommendation with recent rain."""
        from src.infrastructure.weather_service import AgriculturalWeatherData
        
        data = AgriculturalWeatherData(
            location="Test",
            latitude=0.0,
            longitude=0.0,
            temperature=20.0,
            humidity=70.0,
            precipitation=10.0,  # Recent rain
            wind_speed=5.0,
            evapotranspiration=3.0
        )
        
        rec = data.get_irrigation_recommendation()
        assert "No irrigation needed" in rec
    
    def test_irrigation_recommendation_high_et(self):
        """Test recommendation with high evapotranspiration."""
        from src.infrastructure.weather_service import AgriculturalWeatherData
        
        data = AgriculturalWeatherData(
            location="Test",
            latitude=0.0,
            longitude=0.0,
            temperature=30.0,
            humidity=30.0,
            precipitation=0.0,
            wind_speed=15.0,
            evapotranspiration=7.5  # High ET
        )
        
        rec = data.get_irrigation_recommendation()
        assert "High evapotranspiration" in rec
    
    def test_irrigation_recommendation_high_temp(self):
        """Test recommendation with high temperature."""
        from src.infrastructure.weather_service import AgriculturalWeatherData
        
        data = AgriculturalWeatherData(
            location="Test",
            latitude=0.0,
            longitude=0.0,
            temperature=38.0,  # High temp
            humidity=35.0,
            precipitation=0.0,
            wind_speed=10.0,
            evapotranspiration=5.0
        )
        
        rec = data.get_irrigation_recommendation()
        assert "High temperature" in rec


class TestWeatherService:
    """Test WeatherService class."""
    
    def test_service_initialization(self):
        """Test service can be initialized."""
        from src.infrastructure.weather_service import WeatherService
        
        service = WeatherService(cache_ttl=60, timeout=5)
        
        assert service.cache_ttl == 60
        assert service.timeout == 5
    
    def test_weather_code_description(self):
        """Test WMO weather code conversion."""
        from src.infrastructure.weather_service import get_weather_description
        
        assert get_weather_description(0) == "Clear sky"
        assert get_weather_description(61) == "Slight rain"
        assert get_weather_description(95) == "Thunderstorm"
        assert get_weather_description(999) == "Unknown"
    
    @patch('src.infrastructure.weather_service.requests.get')
    def test_geocode_location(self, mock_get):
        """Test location geocoding."""
        from src.infrastructure.weather_service import WeatherService
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{
                "latitude": 31.63,
                "longitude": -8.0,
                "name": "Marrakech",
                "country": "Morocco"
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        service = WeatherService()
        coords = service.geocode_location("Marrakech")
        
        assert coords is not None
        assert coords[0] == 31.63  # latitude
        assert coords[1] == -8.0   # longitude
        assert "Marrakech" in coords[2]
    
    def test_cache_hit(self):
        """Test cache returns cached value."""
        from src.infrastructure.weather_service import WeatherService
        
        service = WeatherService(cache_ttl=3600)
        
        # Manually set cache
        service._set_cache("test_key", "cached_value")
        
        # Should return cached value
        result = service._get_cached("test_key")
        assert result == "cached_value"
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        from src.infrastructure.weather_service import WeatherService
        
        service = WeatherService()
        result = service._get_cached("nonexistent_key")
        
        assert result is None


# ============================================================================
# Weather Agent Tests
# ============================================================================

class TestWeatherAgent:
    """Test WeatherAgent class."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent(relevance_threshold=0.5)
        assert agent.relevance_threshold == 0.5
    
    def test_is_weather_relevant_irrigation(self):
        """Test weather relevance for irrigation queries."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        
        # Should be relevant
        is_relevant, confidence, reasons = agent.is_weather_relevant(
            "When should I irrigate my tomatoes?"
        )
        
        assert is_relevant == True
        assert confidence >= 0.5
        assert len(reasons) > 0
    
    def test_is_weather_relevant_frost(self):
        """Test weather relevance for frost protection queries."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        
        is_relevant, confidence, reasons = agent.is_weather_relevant(
            "How to protect crops from frost damage?"
        )
        
        assert is_relevant == True
        assert confidence >= 0.5
    
    def test_is_weather_relevant_planting(self):
        """Test weather relevance for planting queries."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        
        is_relevant, confidence, reasons = agent.is_weather_relevant(
            "When is the best time to plant wheat?"
        )
        
        assert is_relevant == True
    
    def test_not_weather_relevant_definition(self):
        """Test that definition queries are not weather-relevant."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        
        is_relevant, confidence, reasons = agent.is_weather_relevant(
            "What is drip irrigation?"
        )
        
        # Should NOT be relevant (definition query)
        assert is_relevant == False or confidence < 0.5
    
    def test_not_weather_relevant_comparison(self):
        """Test that comparison queries are not weather-relevant."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        
        is_relevant, confidence, _ = agent.is_weather_relevant(
            "Compare drip and sprinkler irrigation systems"
        )
        
        # Should NOT be relevant or low confidence
        assert is_relevant == False or confidence < 0.5
    
    def test_set_and_get_location(self):
        """Test location management."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        
        assert agent.get_user_location() is None
        assert agent.has_user_location() == False
        
        agent.set_user_location("Casablanca")
        
        assert agent.get_user_location() == "Casablanca"
        assert agent.has_user_location() == True
    
    def test_clear_location(self):
        """Test location clearing."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        agent.set_user_location("Fes")
        agent.clear_user_location()
        
        assert agent.get_user_location() is None
    
    def test_get_weather_context_needs_location(self):
        """Test weather context when location is needed."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        # Don't set location
        
        ctx = agent.get_weather_context("When should I irrigate?")
        
        assert ctx.is_relevant == True
        assert ctx.needs_location == True
    
    def test_format_location_request(self):
        """Test location request formatting."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        request = agent.format_location_request()
        
        assert "location" in request.lower()
        assert "city" in request.lower() or "region" in request.lower()


class TestWeatherContext:
    """Test WeatherContext dataclass."""
    
    def test_context_creation(self):
        """Test WeatherContext creation."""
        from src.agents.weather_agent import WeatherContext
        
        ctx = WeatherContext(
            is_relevant=True,
            confidence=0.8,
            location="Rabat",
            needs_location=False,
            relevance_reasons=["Irrigation timing"]
        )
        
        assert ctx.is_relevant == True
        assert ctx.confidence == 0.8
        assert ctx.location == "Rabat"
    
    def test_context_to_dict(self):
        """Test dictionary conversion."""
        from src.agents.weather_agent import WeatherContext
        
        ctx = WeatherContext(
            is_relevant=True,
            confidence=0.7,
            relevance_reasons=["Test"]
        )
        
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert d["is_relevant"] == True
        assert d["confidence"] == 0.7


# ============================================================================
# Integration Tests (require network - marked slow)
# ============================================================================

@pytest.mark.slow
class TestWeatherIntegration:
    """Integration tests that make real API calls."""
    
    def test_real_geocoding(self):
        """Test real geocoding API call."""
        from src.infrastructure.weather_service import WeatherService
        
        service = WeatherService()
        coords = service.geocode_location("Marrakech")
        
        assert coords is not None
        assert 31 < coords[0] < 32  # Latitude ~31.6
        assert -9 < coords[1] < -7  # Longitude ~-8.0
    
    def test_real_weather_fetch(self):
        """Test real weather API call."""
        from src.infrastructure.weather_service import WeatherService
        
        service = WeatherService()
        data = service.get_agricultural_metrics("Casablanca")
        
        assert data is not None
        assert data.location is not None
        # Temperature should be within reasonable range
        assert -20 < data.temperature < 60
    
    def test_weather_agent_with_location(self):
        """Test complete weather agent flow."""
        from src.agents.weather_agent import WeatherAgent
        
        agent = WeatherAgent()
        agent.set_user_location("Agadir")
        
        ctx = agent.get_weather_context("When should I irrigate my crops?")
        
        assert ctx.is_relevant == True
        assert ctx.needs_location == False
        assert ctx.location is not None
        assert len(ctx.formatted_context) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
