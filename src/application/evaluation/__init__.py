"""
Application - Evaluation Package

Consolidated evaluation modules:
- rag_eval: RAG Triad evaluation (Faithfulness, Relevancy)
- agentic_eval: Agentic metrics (Step Efficiency, PPI Calibration)
- chunking_eval: Chunking quality metrics
"""
from .rag_eval import (
    LLMAsJudgeEvaluator,
    FaithfulnessJudge,
    AnswerRelevancyJudge,
    ContextRelevancyJudge,
    AgriculturalAccuracyJudge,
    SafetyJudge,
    evaluate_rag_response,
)
from .agentic_eval import (
    AgenticEvaluator,
    PPICalibrator,
    create_agentic_evaluator,
)

# Backward compatibility aliases
AgronomistAuditor = LLMAsJudgeEvaluator

__all__ = [
    # RAG evaluation
    "LLMAsJudgeEvaluator",
    "FaithfulnessJudge",
    "AnswerRelevancyJudge", 
    "ContextRelevancyJudge",
    "AgriculturalAccuracyJudge",
    "SafetyJudge",
    "evaluate_rag_response",
    # Agentic evaluation
    "AgenticEvaluator",
    "PPICalibrator",
    "create_agentic_evaluator",
    # Aliases
    "AgronomistAuditor",
]
