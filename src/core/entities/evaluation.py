"""
Core Domain Entities - Evaluation Types

Contains:
- VerdictType: Evaluation verdict enum
- JudgeScore: Score from LLM judge
- EvaluationReport: Complete evaluation report
- AgenticMetrics: Agentic evaluation metrics
- ClaimVerification: Individual claim verification
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional


class VerdictType(Enum):
    """Evaluation verdict types."""
    PASS = "PASS"
    MARGINAL = "MARGINAL"
    FAIL = "FAIL"


class SafetyLevel(Enum):
    """Safety assessment levels."""
    SAFE = "Safe"
    CAUTION = "Caution"
    UNSAFE = "Unsafe"


@dataclass
class JudgeScore:
    """Score from an LLM judge evaluation."""
    metric_name: str
    score: float
    reason: str
    raw_output: str = ""


@dataclass
class ClaimVerification:
    """Individual claim verification result."""
    claim: str
    grounded: bool
    supporting_evidence: str = ""
    confidence: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report from all judges."""
    query: str
    response: str
    context_used: List[str]
    
    # Metric scores
    faithfulness: Optional[JudgeScore] = None
    answer_relevancy: Optional[JudgeScore] = None
    context_relevancy: Optional[JudgeScore] = None
    context_precision: Optional[JudgeScore] = None
    agricultural_accuracy: Optional[JudgeScore] = None
    safety_assessment: Optional[JudgeScore] = None
    
    # Overall
    overall_score: float = 0.0
    verdict: VerdictType = VerdictType.FAIL
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response[:200],
            "scores": {
                "faithfulness": self.faithfulness.score if self.faithfulness else None,
                "answer_relevancy": self.answer_relevancy.score if self.answer_relevancy else None,
                "context_relevancy": self.context_relevancy.score if self.context_relevancy else None,
                "agricultural_accuracy": self.agricultural_accuracy.score if self.agricultural_accuracy else None,
                "safety": self.safety_assessment.score if self.safety_assessment else None,
            },
            "overall_score": self.overall_score,
            "verdict": self.verdict.value,
            "recommendations": self.recommendations,
        }


@dataclass
class AgenticMetrics:
    """
    Complete agentic evaluation metrics.
    Combines RAG Triad with Agentic-specific metrics.
    """
    # RAG Triad (0-1 scale)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0
    
    # Agentic Metrics
    step_efficiency: float = 0.0  # shortest_path / actual_hops
    loop_prevention: float = 1.0  # 1.0 if no loops detected
    tool_selection_accuracy: float = 0.0
    
    # Claim Analysis
    total_claims: int = 0
    grounded_claims: int = 0
    claims: List[ClaimVerification] = field(default_factory=list)
    
    # PPI Calibration
    raw_score: float = 0.0
    calibrated_score: float = 0.0
    ppi_adjustment: float = 0.0
    
    # Overall
    overall_score: float = 0.0
    verdict: str = "FAIL"
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["claims"] = [asdict(c) for c in self.claims]
        return result
    
    def summary(self) -> str:
        return (
            f"Faithfulness: {self.faithfulness:.2f} | "
            f"Step Efficiency: {self.step_efficiency:.2f} | "
            f"Calibrated: {self.calibrated_score:.2f}"
        )
