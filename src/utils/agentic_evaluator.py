"""
Agentic Evaluation Module - ARES + DeepEval 2025 Framework

Implements advanced evaluation metrics for Agentic RAG systems:
- Faithfulness: Ratio of claims grounded in context
- Step Efficiency: Actual hops vs shortest path
- PPI Calibration: Prediction-Powered Inference for LLM-as-Judge calibration
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..config.config import LLM_CONFIG, logger

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ClaimVerification:
    """Individual claim verification result."""
    claim: str
    grounded: bool
    supporting_evidence: str = ""
    confidence: float = 0.0


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
    raw_score: float = 0.0  # Before calibration
    calibrated_score: float = 0.0  # After PPI
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


@dataclass 
class PPICalibrator:
    """
    Prediction-Powered Inference (PPI) Calibrator.
    
    Calibrates LLM-as-Judge scores against a human-verified baseline.
    Uses statistical correction to reduce LLM judge bias.
    """
    human_scores: List[float] = field(default_factory=list)
    llm_scores: List[float] = field(default_factory=list)
    
    # Calibration parameters (learned from baseline)
    bias: float = 0.0
    scale: float = 1.0
    
    def fit(
        self, 
        human_scores: List[float], 
        llm_scores: List[float]
    ):
        """
        Fit calibration parameters from human-verified baseline.
        
        Expects ~50 samples for reliable calibration.
        
        Args:
            human_scores: Ground truth human judgments (0-1)
            llm_scores: Corresponding LLM judge scores (0-1)
        """
        if len(human_scores) < 5 or len(llm_scores) < 5:
            logger.warning("PPI requires more samples for reliable calibration")
            return
        
        self.human_scores = human_scores
        self.llm_scores = llm_scores
        
        # Calculate bias (systematic over/under estimation)
        self.bias = np.mean(llm_scores) - np.mean(human_scores)
        
        # Calculate scale (variance adjustment)
        llm_std = np.std(llm_scores)
        human_std = np.std(human_scores)
        
        if llm_std > 0:
            self.scale = human_std / llm_std
        else:
            self.scale = 1.0
        
        logger.info(f"PPI Calibration: bias={self.bias:.3f}, scale={self.scale:.3f}")
    
    def calibrate(self, llm_score: float) -> float:
        """
        Apply PPI calibration to an LLM score.
        
        Formula: calibrated = (llm_score - bias) * scale
        
        Args:
            llm_score: Raw LLM-as-Judge score (0-1)
            
        Returns:
            Calibrated score (0-1)
        """
        # Remove bias and apply scale
        calibrated = (llm_score - self.bias) * self.scale
        
        # Center around human mean
        if self.human_scores:
            human_mean = np.mean(self.human_scores)
            llm_mean = np.mean(self.llm_scores) if self.llm_scores else 0.5
            calibrated = calibrated + (human_mean - llm_mean * self.scale)
        
        # Clip to valid range
        return float(np.clip(calibrated, 0, 1))
    
    def get_confidence_interval(
        self, 
        llm_score: float, 
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Get confidence interval for calibrated score.
        
        Uses PPI variance estimation.
        
        Args:
            llm_score: Raw LLM score
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        calibrated = self.calibrate(llm_score)
        
        if not self.human_scores or not self.llm_scores:
            return (calibrated - 0.1, calibrated + 0.1)
        
        # Estimate variance from residuals
        residuals = [
            h - self.calibrate(l) 
            for h, l in zip(self.human_scores, self.llm_scores)
        ]
        residual_std = np.std(residuals)
        
        # Z-score for confidence level
        from scipy import stats
        try:
            z = stats.norm.ppf(1 - alpha / 2)
        except:
            z = 1.96  # Fallback for 95% CI
        
        margin = z * residual_std / np.sqrt(len(self.human_scores))
        
        lower = max(0, calibrated - margin)
        upper = min(1, calibrated + margin)
        
        return (lower, upper)
    
    def save(self, filepath: str):
        """Save calibration parameters."""
        data = {
            "bias": self.bias,
            "scale": self.scale,
            "human_scores": self.human_scores,
            "llm_scores": self.llm_scores
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load calibration parameters."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.bias = data.get("bias", 0.0)
            self.scale = data.get("scale", 1.0)
            self.human_scores = data.get("human_scores", [])
            self.llm_scores = data.get("llm_scores", [])
            logger.info(f"Loaded PPI calibration from {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load PPI calibration: {e}")


# ============================================================================
# Agentic Evaluator
# ============================================================================

class AgenticEvaluator:
    """
    Advanced evaluator for Agentic RAG systems.
    
    Implements:
    1. RAG Triad (Faithfulness, Answer Relevancy, Context Relevancy)
    2. Step Efficiency metric
    3. PPI Calibration for LLM-as-Judge scores
    """
    
    def __init__(
        self,
        llm_model: str = None,
        ppi_calibrator: PPICalibrator = None
    ):
        """
        Initialize the Agentic Evaluator.
        
        Args:
            llm_model: LLM model for evaluation
            ppi_calibrator: Pre-fitted PPI calibrator
        """
        self.llm_model = llm_model or LLM_CONFIG.get("model", "gemma2:9b")
        self.ppi_calibrator = ppi_calibrator or PPICalibrator()
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = openai.OpenAI(
                    base_url=LLM_CONFIG.get("base_url", "http://localhost:11434/v1"),
                    api_key=LLM_CONFIG.get("api_key", "ollama")
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        logger.info("Agentic Evaluator initialized")
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: List[str],
        hops_taken: int = 1,
        shortest_path: int = 1,
        entities_visited: List[str] = None,
        apply_ppi: bool = True
    ) -> AgenticMetrics:
        """
        Run full agentic evaluation.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context chunks
            hops_taken: Actual retrieval hops taken
            shortest_path: Theoretical shortest path
            entities_visited: Entities traversed during retrieval
            apply_ppi: Apply PPI calibration
            
        Returns:
            AgenticMetrics with all scores
        """
        metrics = AgenticMetrics()
        
        # Combine context
        context_text = "\n\n".join(context) if context else ""
        
        # 1. Faithfulness (claim verification)
        faithfulness, claims = self._evaluate_faithfulness(answer, context_text)
        metrics.faithfulness = faithfulness
        metrics.claims = claims
        metrics.total_claims = len(claims)
        metrics.grounded_claims = sum(1 for c in claims if c.grounded)
        
        # 2. Answer Relevancy
        metrics.answer_relevancy = self._evaluate_answer_relevancy(query, answer)
        
        # 3. Context Relevancy
        metrics.context_relevancy = self._evaluate_context_relevancy(query, context_text)
        
        # 4. Step Efficiency
        metrics.step_efficiency = self._calculate_step_efficiency(
            hops_taken, 
            shortest_path
        )
        
        # 5. Loop Prevention (check for duplicate entities)
        if entities_visited:
            unique_entities = set(entities_visited)
            metrics.loop_prevention = len(unique_entities) / len(entities_visited) if entities_visited else 1.0
        
        # Calculate raw overall score (RAG Triad average)
        metrics.raw_score = (
            metrics.faithfulness * 0.4 +
            metrics.answer_relevancy * 0.3 +
            metrics.context_relevancy * 0.2 +
            metrics.step_efficiency * 0.1
        )
        
        # 6. Apply PPI Calibration
        if apply_ppi:
            metrics.calibrated_score = self.ppi_calibrator.calibrate(metrics.raw_score)
            metrics.ppi_adjustment = metrics.calibrated_score - metrics.raw_score
        else:
            metrics.calibrated_score = metrics.raw_score
        
        # Overall score and verdict
        metrics.overall_score = metrics.calibrated_score
        
        if metrics.overall_score >= 0.7:
            metrics.verdict = "PASS"
        elif metrics.overall_score >= 0.5:
            metrics.verdict = "MARGINAL"
        else:
            metrics.verdict = "FAIL"
        
        return metrics
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> Tuple[float, List[ClaimVerification]]:
        """
        Evaluate faithfulness via claim extraction and verification.
        
        Faithfulness = |grounded_claims| / |total_claims|
        """
        if not answer or not context:
            return 0.0, []
        
        claims = []
        
        if self.client:
            try:
                claims = self._llm_extract_and_verify_claims(answer, context)
            except Exception as e:
                logger.warning(f"LLM claim verification failed: {e}")
        
        # Fallback to heuristic
        if not claims:
            claims = self._heuristic_claim_verification(answer, context)
        
        if not claims:
            return 0.5, []  # Default if no claims extracted
        
        grounded_count = sum(1 for c in claims if c.grounded)
        faithfulness = grounded_count / len(claims)
        
        return faithfulness, claims
    
    def _llm_extract_and_verify_claims(
        self,
        answer: str,
        context: str
    ) -> List[ClaimVerification]:
        """LLM-based claim extraction and verification."""
        prompt = f"""Extract factual claims from the ANSWER and verify each against the CONTEXT.

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

For each claim, determine if it is GROUNDED (supported by context) or NOT GROUNDED.

Output format (one claim per line):
GROUNDED: [claim text] | EVIDENCE: [quote from context]
NOT_GROUNDED: [claim text]

Extract up to 5 main factual claims."""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        return self._parse_claims(content)
    
    def _parse_claims(self, content: str) -> List[ClaimVerification]:
        """Parse LLM claim verification output."""
        claims = []
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("GROUNDED:"):
                parts = line[9:].split("|")
                claim_text = parts[0].strip()
                evidence = parts[1].replace("EVIDENCE:", "").strip() if len(parts) > 1 else ""
                claims.append(ClaimVerification(
                    claim=claim_text,
                    grounded=True,
                    supporting_evidence=evidence,
                    confidence=0.9
                ))
            elif line.startswith("NOT_GROUNDED:"):
                claim_text = line[13:].strip()
                claims.append(ClaimVerification(
                    claim=claim_text,
                    grounded=False,
                    confidence=0.9
                ))
        
        return claims
    
    def _heuristic_claim_verification(
        self,
        answer: str,
        context: str
    ) -> List[ClaimVerification]:
        """Simple heuristic claim verification."""
        claims = []
        
        # Split answer into sentences as claims
        sentences = answer.split(". ")
        context_lower = context.lower()
        
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check for keyword overlap
            words = set(sentence.lower().split())
            # Remove common words
            common = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "and", "or", "for", "to", "in", "on", "at", "by", "with"}
            meaningful_words = words - common
            
            overlap = sum(1 for w in meaningful_words if w in context_lower)
            grounded = overlap >= len(meaningful_words) * 0.3
            
            claims.append(ClaimVerification(
                claim=sentence[:100],
                grounded=grounded,
                confidence=0.6
            ))
        
        return claims
    
    def _evaluate_answer_relevancy(
        self,
        query: str,
        answer: str
    ) -> float:
        """Evaluate if answer is relevant to query."""
        if not answer or not query:
            return 0.0
        
        if self.client:
            try:
                prompt = f"""Rate how well the ANSWER addresses the QUESTION.
                
QUESTION: {query}

ANSWER: {answer}

Score from 0-10 where 10 = perfectly addresses the question.
Only output the number."""

                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text.split()[0]) / 10
                return min(1.0, max(0.0, score))
            except:
                pass
        
        # Heuristic fallback
        query_words = set(query.lower().split()) - {"what", "how", "when", "where", "why", "is", "the", "a", "an"}
        answer_lower = answer.lower()
        overlap = sum(1 for w in query_words if w in answer_lower)
        return min(1.0, overlap / max(len(query_words), 1))
    
    def _evaluate_context_relevancy(
        self,
        query: str,
        context: str
    ) -> float:
        """Evaluate if retrieved context is relevant to query."""
        if not context or not query:
            return 0.0
        
        if self.client:
            try:
                prompt = f"""Rate how relevant the CONTEXT is for answering the QUESTION.
                
QUESTION: {query}

CONTEXT: {context[:2000]}

Score from 0-10 where 10 = highly relevant context.
Only output the number."""

                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text.split()[0]) / 10
                return min(1.0, max(0.0, score))
            except:
                pass
        
        # Heuristic fallback
        query_words = set(query.lower().split()) - {"what", "how", "when", "where", "why", "is", "the", "a", "an"}
        context_lower = context.lower()
        overlap = sum(1 for w in query_words if w in context_lower)
        return min(1.0, overlap / max(len(query_words), 1))
    
    def _calculate_step_efficiency(
        self,
        hops_taken: int,
        shortest_path: int
    ) -> float:
        """
        Calculate step efficiency metric.
        
        Step Efficiency = shortest_path / actual_hops
        1.0 = optimal (found answer in minimum hops)
        <1.0 = took extra steps
        """
        if hops_taken <= 0:
            return 1.0
        if shortest_path <= 0:
            shortest_path = 1
        
        efficiency = shortest_path / hops_taken
        return min(1.0, efficiency)
    
    def calibrate_with_baseline(
        self,
        human_scores: List[float],
        llm_scores: List[float]
    ):
        """
        Fit PPI calibrator with human-verified baseline.
        
        Expects ~50 samples for reliable calibration.
        
        Args:
            human_scores: Ground truth human judgments (0-1)
            llm_scores: Corresponding LLM judge scores (0-1)
        """
        self.ppi_calibrator.fit(human_scores, llm_scores)
    
    def quick_evaluate(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Quick evaluation without full claim extraction.
        
        Returns dictionary of scores.
        """
        metrics = self.evaluate(
            query=query,
            answer=answer,
            context=[context] if context else [],
            hops_taken=1,
            shortest_path=1,
            apply_ppi=True
        )
        
        return {
            "faithfulness": metrics.faithfulness,
            "answer_relevancy": metrics.answer_relevancy,
            "context_relevancy": metrics.context_relevancy,
            "step_efficiency": metrics.step_efficiency,
            "overall_score": metrics.overall_score,
            "verdict": metrics.verdict
        }


# ============================================================================
# Integration with existing evaluator
# ============================================================================

def create_agentic_evaluator(ppi_filepath: str = None) -> AgenticEvaluator:
    """
    Factory function to create an Agentic Evaluator.
    
    Args:
        ppi_filepath: Path to saved PPI calibration parameters
        
    Returns:
        Configured AgenticEvaluator instance
    """
    evaluator = AgenticEvaluator()
    
    if ppi_filepath:
        evaluator.ppi_calibrator.load(ppi_filepath)
    
    return evaluator


def main():
    """Test Agentic Evaluator."""
    evaluator = AgenticEvaluator()
    
    # Test evaluation
    test_query = "How does drip irrigation affect water efficiency?"
    test_answer = "Drip irrigation improves water efficiency by delivering water directly to plant roots, reducing evaporation losses by up to 40%."
    test_context = [
        "Drip irrigation is a method of delivering water directly to the root zone of plants.",
        "Studies show drip irrigation can reduce water usage by 30-50% compared to flood irrigation.",
        "Evaporation losses are minimized in drip systems as water is applied below the soil surface."
    ]
    
    print("Testing Agentic Evaluator...")
    print(f"Query: {test_query}")
    print(f"Answer: {test_answer}")
    print("-" * 60)
    
    metrics = evaluator.evaluate(
        query=test_query,
        answer=test_answer,
        context=test_context,
        hops_taken=2,
        shortest_path=1
    )
    
    print(f"Faithfulness: {metrics.faithfulness:.2f}")
    print(f"Answer Relevancy: {metrics.answer_relevancy:.2f}")
    print(f"Context Relevancy: {metrics.context_relevancy:.2f}")
    print(f"Step Efficiency: {metrics.step_efficiency:.2f}")
    print(f"Overall Score: {metrics.overall_score:.2f}")
    print(f"Verdict: {metrics.verdict}")
    
    print("\nClaim Analysis:")
    for claim in metrics.claims:
        status = "[OK]" if claim.grounded else "[X]"
        print(f"  {status} {claim.claim[:60]}...")
    
    # Test PPI Calibration
    print("\n" + "=" * 60)
    print("Testing PPI Calibration...")
    
    # Simulate human baseline
    human_scores = [0.7, 0.8, 0.6, 0.9, 0.5, 0.75, 0.85, 0.65, 0.7, 0.8]
    llm_scores = [0.8, 0.9, 0.75, 0.95, 0.7, 0.85, 0.9, 0.8, 0.85, 0.9]  # LLM tends to overestimate
    
    evaluator.calibrate_with_baseline(human_scores, llm_scores)
    
    test_llm_score = 0.85
    calibrated = evaluator.ppi_calibrator.calibrate(test_llm_score)
    print(f"Raw LLM Score: {test_llm_score:.2f}")
    print(f"Calibrated Score: {calibrated:.2f}")
    print(f"Bias: {evaluator.ppi_calibrator.bias:.3f}")
    print(f"Scale: {evaluator.ppi_calibrator.scale:.3f}")


if __name__ == "__main__":
    main()
