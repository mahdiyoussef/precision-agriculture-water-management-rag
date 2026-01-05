"""
Weather Agent Module

Intelligent agent that detects when weather context is relevant for
agriculture and water management queries, and enriches them with
real-time weather data.

Features:
- Query relevance detection using pattern matching
- Location management (ask user, cache in session)
- Weather context formatting for LLM
- Integration with RAG orchestrator
"""
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from ..config.config import logger
from ..infrastructure.weather_service import (
    WeatherService,
    AgriculturalWeatherData,
    get_weather_service
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WeatherContext:
    """Weather context for LLM enrichment."""
    is_relevant: bool
    confidence: float
    location: Optional[str] = None
    weather_data: Optional[AgriculturalWeatherData] = None
    formatted_context: str = ""
    needs_location: bool = False
    relevance_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_relevant": self.is_relevant,
            "confidence": self.confidence,
            "location": self.location,
            "needs_location": self.needs_location,
            "formatted_context": self.formatted_context,
            "relevance_reasons": self.relevance_reasons,
            "weather_data": self.weather_data.to_dict() if self.weather_data else None
        }


# =============================================================================
# Weather Relevance Patterns
# =============================================================================

# Patterns that indicate weather context would be helpful
WEATHER_RELEVANCE_PATTERNS = {
    # Irrigation timing
    "irrigation_timing": [
        r'\b(when|best time|schedule)\b.*\b(irrigat|water)\b',
        r'\b(irrigat|water)\b.*\b(when|best time|schedule|timing)\b',
        r'\b(should\s+i|can\s+i)\s+(irrigate|water)\b',
        r'\b(irrigation|watering)\s+(schedule|plan|timing)\b',
        r'\bwater\s+(my|the)\s+\w+\b',  # "water my garden"
    ],
    
    # Temperature-sensitive operations
    "temperature_sensitive": [
        r'\b(frost|freeze|freezing|cold)\s*(protection|damage|risk|warning)?\b',
        r'\b(heat|hot)\s*(stress|wave|damage)?\b',
        r'\b(protect|protection)\b.*(frost|cold|heat|freeze)\b',
        r'\b(temperature|temp)\s+(for|requirement|optimal)\b',
        r'\b(frost|cold)\b',  # Simple frost/cold mentions
    ],
    
    # Planting and harvesting
    "planting_decisions": [
        r'\b(when|best time|good time)\b.*(plant|sow|seed|transplant|harvest)\b',
        r'\b(plant|sow|seed|harvest)\b.*\b(when|time|season)\b',
        r'\b(planting|sowing|seeding|harvesting)\s+(time|season|window|date)\b',
        r'\b(can\s+i|should\s+i)\s+(plant|sow|harvest)\b',
        r'\b(is\s+it\s+)?(a\s+)?good\s+time\s+to\s+(plant|sow|harvest)\b',  # "Is it a good time to plant"
    ],
    
    # Water management
    "water_management": [
        r'\b(drought|dry)\s*(stress|condition|period|management)?\b',
        r'\b(rain|rainfall|precipitation)\s*(forecast|expected|prediction)?\b',
        r'\b(water)\s+(availability|shortage|supply|requirement)\b',
        r'\b(soil\s+moisture|evapotranspiration|ET0|ET₀)\b',
    ],
    
    # Crop stress
    "crop_stress": [
        r'\b(crop|plant)\s+(stress|health|condition)\b',
        r'\b(water|moisture)\s+(stress|deficit)\b',
        r'\b(wilting|wilt|drooping)\b',
    ],
    
    # Explicit weather queries
    "explicit_weather": [
        r'\b(weather|climate)\s*(condition|forecast|today|week|current)?\b',
        r'\b(current|today|this week)\b.*\b(weather|temperature|humidity)\b',
        r'\bweather\s+(in|for|at)\b',
        r'\bforecast\b',  # Simple forecast mention
    ],
    
    # Agricultural operations affected by weather
    "weather_affected_ops": [
        r'\b(spray|spraying|pesticide|herbicide|fungicide)\b.*\b(when|timing|apply)\b',
        r'\b(fertiliz|fertigation)\b.*\b(when|timing|apply)\b',
        r'\b(cover\s+crop|mulch)\b',
    ],
}


# Keywords that suggest weather relevance (lower weight)
WEATHER_KEYWORDS = [
    "irrigation", "irrigate", "watering", "water schedule",
    "frost", "freeze", "cold", "heat", "temperature",
    "drought", "rain", "precipitation", "weather",
    "planting time", "sowing", "harvest", "season",
    "evapotranspiration", "ET0", "humidity", "crop stress",
    "soil moisture", "spray timing", "fertilizer timing"
]

# Topics that are generally NOT weather-dependent
NON_WEATHER_PATTERNS = [
    r'^what\s+is\s+',  # Definitions
    r'^define\s+',
    r'^explain\s+(what|the\s+concept)\b',
    r'\b(history|origin|invention)\s+of\b',
    r'\b(compare|comparison|difference)\s+between\b',
    r'\b(types|kinds|categories)\s+of\b',
]


# =============================================================================
# Weather Agent
# =============================================================================

class WeatherAgent:
    """
    Intelligent agent that detects weather-relevant queries and
    enriches them with real-time weather context.
    
    Workflow:
    1. Analyze query for weather relevance
    2. If relevant and location unknown → signal needs_location
    3. If location known → fetch weather and format for LLM
    """
    
    def __init__(
        self,
        weather_service: Optional[WeatherService] = None,
        relevance_threshold: float = 0.5
    ):
        """
        Initialize the weather agent.
        
        Args:
            weather_service: WeatherService instance (uses singleton if None)
            relevance_threshold: Minimum confidence to consider relevant
        """
        self.weather_service = weather_service or get_weather_service()
        self.relevance_threshold = relevance_threshold
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in WEATHER_RELEVANCE_PATTERNS.items()
        }
        
        self.non_weather_patterns = [
            re.compile(p, re.IGNORECASE) for p in NON_WEATHER_PATTERNS
        ]
        
        # User location cache (per-session)
        self._user_location: Optional[str] = None
        
        logger.info("WeatherAgent initialized")
    
    # =========================================================================
    # Location Management
    # =========================================================================
    
    def set_user_location(self, location: str):
        """Set the user's location for the session."""
        self._user_location = location
        logger.info(f"User location set to: {location}")
    
    def get_user_location(self) -> Optional[str]:
        """Get the user's cached location."""
        return self._user_location
    
    def clear_user_location(self):
        """Clear the user's cached location."""
        self._user_location = None
        logger.info("User location cleared")
    
    def has_user_location(self) -> bool:
        """Check if user location is set."""
        return self._user_location is not None
    
    # =========================================================================
    # Query Analysis
    # =========================================================================
    
    def is_weather_relevant(self, query: str) -> Tuple[bool, float, List[str]]:
        """
        Determine if a query would benefit from weather context.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (is_relevant, confidence, reasons)
        """
        query_lower = query.lower().strip()
        
        # Check for non-weather patterns (early exit)
        for pattern in self.non_weather_patterns:
            if pattern.match(query_lower):
                return False, 0.1, []
        
        score = 0.0
        reasons = []
        
        # Pattern matching (high weight)
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    score += 0.4
                    reasons.append(category.replace("_", " ").title())
                    break  # Only count each category once
        
        # Keyword matching (medium weight)
        keyword_matches = []
        for keyword in WEATHER_KEYWORDS:
            if keyword.lower() in query_lower:
                score += 0.1
                keyword_matches.append(keyword)
        
        if keyword_matches:
            reasons.append(f"Keywords: {', '.join(keyword_matches[:3])}")
        
        # Cap at 1.0
        confidence = min(score, 1.0)
        is_relevant = confidence >= self.relevance_threshold
        
        logger.debug(f"Weather relevance: {is_relevant} (conf: {confidence:.2f})")
        return is_relevant, confidence, reasons
    
    # =========================================================================
    # Weather Context Generation
    # =========================================================================
    
    def get_weather_context(
        self,
        query: str,
        location: Optional[str] = None
    ) -> WeatherContext:
        """
        Get weather context for a query.
        
        Args:
            query: User query
            location: Override location (uses cached if None)
            
        Returns:
            WeatherContext object
        """
        # Check relevance
        is_relevant, confidence, reasons = self.is_weather_relevant(query)
        
        if not is_relevant:
            return WeatherContext(
                is_relevant=False,
                confidence=confidence,
                relevance_reasons=[]
            )
        
        # Determine location
        effective_location = location or self._user_location
        
        if not effective_location:
            return WeatherContext(
                is_relevant=True,
                confidence=confidence,
                needs_location=True,
                relevance_reasons=reasons
            )
        
        # Fetch weather data
        weather_data = self.weather_service.get_agricultural_metrics(effective_location)
        
        if not weather_data:
            logger.warning(f"Failed to fetch weather for: {effective_location}")
            return WeatherContext(
                is_relevant=True,
                confidence=confidence,
                location=effective_location,
                needs_location=False,
                relevance_reasons=reasons,
                formatted_context=f"[Weather data unavailable for {effective_location}]"
            )
        
        # Format for LLM
        formatted = self.weather_service.format_for_llm(weather_data)
        
        return WeatherContext(
            is_relevant=True,
            confidence=confidence,
            location=effective_location,
            weather_data=weather_data,
            formatted_context=formatted,
            needs_location=False,
            relevance_reasons=reasons
        )
    
    def format_location_request(self) -> str:
        """Format a request for user's location."""
        return (
            "To provide accurate weather-based recommendations for your query, "
            "I need to know your location. Please provide your city or region name."
        )
    
    # =========================================================================
    # LLM Integration
    # =========================================================================
    
    def enrich_query_context(
        self,
        query: str,
        existing_context: str,
        location: Optional[str] = None
    ) -> Tuple[str, WeatherContext]:
        """
        Enrich existing context with weather data if relevant.
        
        Args:
            query: User query
            existing_context: Context from RAG retrieval
            location: User location (optional)
            
        Returns:
            Tuple of (enriched_context, weather_context)
        """
        weather_ctx = self.get_weather_context(query, location)
        
        if not weather_ctx.is_relevant or weather_ctx.needs_location:
            return existing_context, weather_ctx
        
        if weather_ctx.formatted_context:
            enriched = (
                f"{weather_ctx.formatted_context}\n\n"
                f"{'=' * 50}\n\n"
                f"{existing_context}"
            )
            return enriched, weather_ctx
        
        return existing_context, weather_ctx
    
    def get_weather_prompt_addition(
        self,
        weather_ctx: WeatherContext
    ) -> str:
        """
        Generate additional prompt instructions for weather-aware response.
        
        Args:
            weather_ctx: Weather context object
            
        Returns:
            Additional prompt text
        """
        if not weather_ctx.is_relevant or not weather_ctx.weather_data:
            return ""
        
        return (
            "\n\nIMPORTANT: Weather context has been provided above. "
            "Consider current weather conditions, forecasts, and agricultural metrics "
            "(such as evapotranspiration and soil moisture) when providing your recommendations. "
            "Reference specific weather data in your response when relevant."
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_weather_agent(
    weather_service: Optional[WeatherService] = None,
    relevance_threshold: float = 0.5
) -> WeatherAgent:
    """
    Create a WeatherAgent instance.
    
    Args:
        weather_service: Optional WeatherService instance
        relevance_threshold: Minimum relevance confidence
        
    Returns:
        Configured WeatherAgent
    """
    return WeatherAgent(
        weather_service=weather_service,
        relevance_threshold=relevance_threshold
    )


# =============================================================================
# Main (Testing)
# =============================================================================

def main():
    """Test the weather agent."""
    print("Weather Agent Test")
    print("=" * 60)
    
    agent = WeatherAgent()
    
    # Test queries
    test_queries = [
        # Should be weather-relevant
        "When should I irrigate my tomatoes?",
        "Is it a good time to plant wheat?",
        "How to protect crops from frost?",
        "What's the weather forecast for crop planning?",
        "Should I water my garden today?",
        "When is the best time to harvest olives?",
        
        # Should NOT be weather-relevant
        "What is drip irrigation?",
        "Compare drip and sprinkler irrigation",
        "History of precision agriculture",
        "Types of soil moisture sensors",
    ]
    
    print("\nQuery Relevance Analysis:")
    print("-" * 60)
    
    for query in test_queries:
        is_relevant, confidence, reasons = agent.is_weather_relevant(query)
        status = "✓ RELEVANT" if is_relevant else "✗ Not relevant"
        print(f"\n{status} (conf: {confidence:.2f})")
        print(f"  Query: {query}")
        if reasons:
            print(f"  Reasons: {', '.join(reasons)}")
    
    # Test with location
    print("\n" + "=" * 60)
    print("\nWeather Context Generation:")
    print("-" * 60)
    
    agent.set_user_location("Marrakech")
    
    ctx = agent.get_weather_context("When should I irrigate my tomatoes?")
    print(f"\nLocation: {ctx.location}")
    print(f"Relevant: {ctx.is_relevant}")
    print(f"Needs Location: {ctx.needs_location}")
    if ctx.formatted_context:
        print(f"\nFormatted Context:\n{ctx.formatted_context[:500]}...")


if __name__ == "__main__":
    main()
