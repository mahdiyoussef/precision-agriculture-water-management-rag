"""
LLM-as-Judge Evaluation Module
Implements multiple evaluation frameworks:
- DeepEval: Comprehensive RAG metrics
- Custom LLM Judge: Agricultural domain-specific evaluation
- RAGAS-style metrics: Faithfulness, Relevancy, Context metrics

Supports local Ollama models as judges for privacy and cost-efficiency.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from langchain_community.llms import Ollama

from src.config.config import LLM_CONFIG, logger


# ============================================================================
# Base Classes and Data Structures
# ============================================================================

class VerdictType(Enum):
    PASS = "PASS"
    MARGINAL = "MARGINAL"
    FAIL = "FAIL"


class SafetyLevel(Enum):
    SAFE = "Safe"
    CAUTION = "Caution"
    UNSAFE = "Unsafe"


@dataclass
class JudgeScore:
    """Score from an LLM judge evaluation."""
    metric_name: str
    score: float  # 0.0 to 1.0
    reason: str
    raw_output: str = ""


@dataclass 
class EvaluationReport:
    """Complete evaluation report from all judges."""
    query: str
    response: str
    context_used: List[str]
    
    # Individual metric scores
    faithfulness: JudgeScore = None
    answer_relevancy: JudgeScore = None
    context_relevancy: JudgeScore = None
    context_precision: JudgeScore = None
    context_recall: JudgeScore = None
    
    # Agricultural-specific scores
    agricultural_accuracy: JudgeScore = None
    safety_assessment: JudgeScore = None
    
    # Aggregate
    overall_score: float = 0.0
    verdict: VerdictType = VerdictType.FAIL
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        scores = {}
        for attr in ['faithfulness', 'answer_relevancy', 'context_relevancy', 
                     'context_precision', 'context_recall', 'agricultural_accuracy',
                     'safety_assessment']:
            score = getattr(self, attr)
            if score:
                scores[attr] = {
                    "score": round(score.score, 3),
                    "reason": score.reason
                }
        
        return {
            "query": self.query[:100],
            "scores": scores,
            "overall_score": round(self.overall_score, 3),
            "verdict": self.verdict.value,
            "recommendations": self.recommendations
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# LLM Judge Base Class
# ============================================================================

class LLMJudge(ABC):
    """Base class for LLM-based evaluation judges."""
    
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or LLM_CONFIG["model"]
        self.base_url = base_url or LLM_CONFIG["base_url"]
        self.llm = Ollama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.0,  # Deterministic for consistent evaluation
            num_ctx=4096,
        )
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> JudgeScore:
        """Evaluate and return a score."""
        pass
    
    def _parse_score(self, output: str, default: float = 0.5) -> float:
        """Parse a numeric score from LLM output."""
        # Look for score patterns
        patterns = [
            r'(?:score|rating)[:\s]*(\d+(?:\.\d+)?)\s*(?:/\s*10|/\s*5|/\s*1)?',
            r'(\d+(?:\.\d+)?)\s*(?:/\s*10|/\s*5|out of)',
            r'(?:^|\n)\s*(\d+(?:\.\d+)?)\s*(?:$|\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1
                if score > 1:
                    if score <= 5:
                        score = score / 5
                    elif score <= 10:
                        score = score / 10
                    else:
                        score = min(1.0, score / 100)
                return min(1.0, max(0.0, score))
        
        return default
    
    def _parse_verdict(self, output: str) -> str:
        """Parse a verdict from LLM output."""
        output_lower = output.lower()
        if 'yes' in output_lower or 'pass' in output_lower or 'true' in output_lower:
            return 'yes'
        elif 'no' in output_lower or 'fail' in output_lower or 'false' in output_lower:
            return 'no'
        return 'unclear'


# ============================================================================
# Specific Metric Judges
# ============================================================================

class FaithfulnessJudge(LLMJudge):
    """
    Evaluates if the response is faithful to the retrieved context.
    Checks for hallucinations and unsupported claims.
    """
    
    PROMPT = """You are an expert fact-checker. Your task is to evaluate if the ANSWER is faithful to the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Evaluation criteria:
1. Every claim in the ANSWER must be supported by the CONTEXT
2. No information should be fabricated or hallucinated
3. Numbers, units, and technical terms must match exactly

For each statement in the answer, check if it's supported by context.

Provide your evaluation in this format:
SUPPORTED_CLAIMS: [list claims that are in context]
UNSUPPORTED_CLAIMS: [list claims not in context]
SCORE: [0-10 where 10 = fully faithful]
REASON: [brief explanation]"""

    def evaluate(self, answer: str, context: str) -> JudgeScore:
        prompt = self.PROMPT.format(context=context, answer=answer)
        
        try:
            output = self.llm.invoke(prompt)
            score = self._parse_score(output)
            
            # Extract reason
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', output, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Evaluation completed"
            
            return JudgeScore(
                metric_name="faithfulness",
                score=score,
                reason=reason[:200],
                raw_output=output
            )
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return JudgeScore("faithfulness", 0.5, f"Evaluation error: {str(e)}")


class AnswerRelevancyJudge(LLMJudge):
    """
    Evaluates if the answer is relevant to the question.
    """
    
    PROMPT = """You are evaluating answer quality. Does the ANSWER directly address the QUESTION?

QUESTION: {question}

ANSWER: {answer}

Evaluate on these criteria:
1. Does the answer address the specific question asked?
2. Is the answer direct and not evasive?
3. Does it provide actionable information?
4. Is it appropriate for a farmer/agronomist audience?

SCORE: [0-10 where 10 = perfectly relevant]
REASON: [why this score]"""

    def evaluate(self, question: str, answer: str) -> JudgeScore:
        prompt = self.PROMPT.format(question=question, answer=answer)
        
        try:
            output = self.llm.invoke(prompt)
            score = self._parse_score(output)
            
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', output, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Evaluation completed"
            
            return JudgeScore(
                metric_name="answer_relevancy",
                score=score,
                reason=reason[:200],
                raw_output=output
            )
        except Exception as e:
            logger.error(f"Answer relevancy evaluation failed: {e}")
            return JudgeScore("answer_relevancy", 0.5, f"Evaluation error: {str(e)}")


class ContextRelevancyJudge(LLMJudge):
    """
    Evaluates if the retrieved context is relevant to the question.
    """
    
    PROMPT = """Evaluate if the CONTEXT is relevant for answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}

Criteria:
1. Does the context contain information needed to answer the question?
2. Is the context specific to the topic asked about?
3. Are the key concepts from the question present in the context?

SCORE: [0-10 where 10 = highly relevant context]
RELEVANT_PARTS: [key relevant snippets]
IRRELEVANT_PARTS: [parts that don't help]
REASON: [explanation]"""

    def evaluate(self, question: str, context: str) -> JudgeScore:
        prompt = self.PROMPT.format(question=question, context=context)
        
        try:
            output = self.llm.invoke(prompt)
            score = self._parse_score(output)
            
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', output, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Evaluation completed"
            
            return JudgeScore(
                metric_name="context_relevancy",
                score=score,
                reason=reason[:200],
                raw_output=output
            )
        except Exception as e:
            logger.error(f"Context relevancy evaluation failed: {e}")
            return JudgeScore("context_relevancy", 0.5, f"Evaluation error: {str(e)}")


class AgriculturalAccuracyJudge(LLMJudge):
    """
    Agricultural domain expert judge.
    Evaluates technical accuracy for precision agriculture and water management.
    """
    
    PROMPT = """You are a Senior Agronomist and Irrigation Engineer. Evaluate the technical accuracy of this agricultural advice.

QUESTION: {question}

ANSWER: {answer}

RETRIEVED CONTEXT: {context}

Evaluate on these agricultural criteria:
1. NUMERICAL ACCURACY: Are all measurements, percentages, and values correct?
2. TECHNICAL TERMS: Are irrigation, soil, and crop terms used correctly?
3. PRACTICAL APPLICABILITY: Is this advice actionable for a farmer?
4. SAFETY: Could following this advice cause crop damage or water waste?
5. REGIONAL APPROPRIATENESS: Is advice suitable for general agricultural practice?

Provide:
NUMERICAL_ISSUES: [list any incorrect numbers or units]
TECHNICAL_ISSUES: [list any misused terms]
SAFETY_CONCERNS: [list any dangerous recommendations]
SCORE: [0-10 for overall agricultural accuracy]
REASON: [explanation as an expert]"""

    def evaluate(self, question: str, answer: str, context: str) -> JudgeScore:
        prompt = self.PROMPT.format(
            question=question, 
            answer=answer, 
            context=context
        )
        
        try:
            output = self.llm.invoke(prompt)
            score = self._parse_score(output)
            
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', output, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Agricultural evaluation completed"
            
            return JudgeScore(
                metric_name="agricultural_accuracy",
                score=score,
                reason=reason[:200],
                raw_output=output
            )
        except Exception as e:
            logger.error(f"Agricultural accuracy evaluation failed: {e}")
            return JudgeScore("agricultural_accuracy", 0.5, f"Evaluation error: {str(e)}")


class SafetyJudge(LLMJudge):
    """
    Evaluates the safety of agricultural recommendations.
    """
    
    PROMPT = """You are an agricultural safety auditor. Evaluate if this advice is SAFE to follow.

ADVICE: {answer}

Critical safety criteria for agriculture:
1. No incorrect irrigation volumes that could flood or drought crops
2. No wrong chemical concentrations or application rates
3. No advice that could lead to soil degradation
4. No recommendations that ignore obvious safety factors
5. No overconfident claims without caveats

Is this advice SAFE to follow? 

SAFETY_ISSUES: [list specific concerns]
VERDICT: [SAFE / CAUTION / UNSAFE]
SCORE: [0-10 where 10 = completely safe]
REASON: [explanation]"""

    def evaluate(self, answer: str) -> JudgeScore:
        prompt = self.PROMPT.format(answer=answer)
        
        try:
            output = self.llm.invoke(prompt)
            score = self._parse_score(output)
            
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', output, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Safety evaluation completed"
            
            return JudgeScore(
                metric_name="safety_assessment",
                score=score,
                reason=reason[:200],
                raw_output=output
            )
        except Exception as e:
            logger.error(f"Safety evaluation failed: {e}")
            return JudgeScore("safety_assessment", 0.5, f"Evaluation error: {str(e)}")


# ============================================================================
# Main Evaluator Class
# ============================================================================

class LLMAsJudgeEvaluator:
    """
    Comprehensive LLM-as-Judge evaluator for RAG systems.
    Uses multiple specialized judges for different aspects.
    """
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize all judge instances."""
        self.model = model or LLM_CONFIG["model"]
        self.base_url = base_url or LLM_CONFIG["base_url"]
        
        # Initialize judges
        self.faithfulness_judge = FaithfulnessJudge(self.model, self.base_url)
        self.relevancy_judge = AnswerRelevancyJudge(self.model, self.base_url)
        self.context_judge = ContextRelevancyJudge(self.model, self.base_url)
        self.agri_judge = AgriculturalAccuracyJudge(self.model, self.base_url)
        self.safety_judge = SafetyJudge(self.model, self.base_url)
        
        logger.info(f"LLM-as-Judge Evaluator initialized with model: {self.model}")
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        run_all: bool = True,
        metrics: List[str] = None
    ) -> EvaluationReport:
        """
        Run full evaluation pipeline.
        
        Args:
            question: User query
            answer: Generated response
            context_chunks: Retrieved context chunks
            run_all: Run all metrics or just specified ones
            metrics: List of specific metrics to run
        
        Returns:
            EvaluationReport with all scores
        """
        # Combine context
        context_texts = [
            c.get('text', c.get('content', ''))
            for c in context_chunks
        ]
        context = "\n\n---\n\n".join(context_texts)
        
        report = EvaluationReport(
            query=question,
            response=answer,
            context_used=context_texts[:3]  # Store first 3 for reference
        )
        
        metrics_to_run = metrics or [
            'faithfulness', 'answer_relevancy', 'context_relevancy',
            'agricultural_accuracy', 'safety_assessment'
        ]
        
        # Run each judge
        if 'faithfulness' in metrics_to_run:
            report.faithfulness = self.faithfulness_judge.evaluate(answer, context)
            logger.info(f"Faithfulness: {report.faithfulness.score:.2f}")
        
        if 'answer_relevancy' in metrics_to_run:
            report.answer_relevancy = self.relevancy_judge.evaluate(question, answer)
            logger.info(f"Answer Relevancy: {report.answer_relevancy.score:.2f}")
        
        if 'context_relevancy' in metrics_to_run:
            report.context_relevancy = self.context_judge.evaluate(question, context)
            logger.info(f"Context Relevancy: {report.context_relevancy.score:.2f}")
        
        if 'agricultural_accuracy' in metrics_to_run:
            report.agricultural_accuracy = self.agri_judge.evaluate(question, answer, context)
            logger.info(f"Agricultural Accuracy: {report.agricultural_accuracy.score:.2f}")
        
        if 'safety_assessment' in metrics_to_run:
            report.safety_assessment = self.safety_judge.evaluate(answer)
            logger.info(f"Safety: {report.safety_assessment.score:.2f}")
        
        # Calculate overall score (weighted average)
        scores = []
        weights = {
            'faithfulness': 0.25,
            'answer_relevancy': 0.20,
            'context_relevancy': 0.15,
            'agricultural_accuracy': 0.25,
            'safety_assessment': 0.15
        }
        
        for attr, weight in weights.items():
            score_obj = getattr(report, attr)
            if score_obj:
                scores.append((score_obj.score, weight))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            report.overall_score = sum(s * w for s, w in scores) / total_weight
        
        # Determine verdict
        if report.overall_score >= 0.7:
            report.verdict = VerdictType.PASS
        elif report.overall_score >= 0.5:
            report.verdict = VerdictType.MARGINAL
        else:
            report.verdict = VerdictType.FAIL
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: EvaluationReport) -> List[str]:
        """Generate actionable recommendations based on scores."""
        recs = []
        
        if report.faithfulness and report.faithfulness.score < 0.6:
            recs.append("Improve grounding: Constrain LLM to only use retrieved context")
        
        if report.answer_relevancy and report.answer_relevancy.score < 0.6:
            recs.append("Improve relevancy: Add explicit instruction to address the question directly")
        
        if report.context_relevancy and report.context_relevancy.score < 0.6:
            recs.append("Improve retrieval: Adjust chunk size or embedding model for better context")
        
        if report.agricultural_accuracy and report.agricultural_accuracy.score < 0.6:
            recs.append("Improve domain accuracy: Fine-tune prompts for agricultural terminology")
        
        if report.safety_assessment and report.safety_assessment.score < 0.7:
            recs.append("Add safety checks: Include warnings for numerical recommendations")
        
        return recs if recs else ["Performance is adequate. Consider fine-tuning for edge cases."]
    
    def quick_evaluate(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Quick evaluation with minimal overhead.
        Returns just the scores without full report.
        """
        faith = self.faithfulness_judge.evaluate(answer, context)
        relevancy = self.relevancy_judge.evaluate(question, answer)
        
        return {
            "faithfulness": faith.score,
            "answer_relevancy": relevancy.score,
            "overall": (faith.score + relevancy.score) / 2
        }


# ============================================================================
# DeepEval Integration (when available)
# ============================================================================

def create_deepeval_evaluator():
    """
    Create a DeepEval-based evaluator if the library is available.
    Falls back to custom LLM judges if not installed.
    """
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric
        )
        from deepeval.test_case import LLMTestCase
        
        class DeepEvalEvaluator:
            """Wrapper for DeepEval metrics."""
            
            def __init__(self):
                self.faithfulness = FaithfulnessMetric()
                self.relevancy = AnswerRelevancyMetric()
                self.context_relevancy = ContextualRelevancyMetric()
                self.precision = ContextualPrecisionMetric()
                self.recall = ContextualRecallMetric()
            
            def evaluate(
                self,
                question: str,
                answer: str,
                context: List[str],
                expected: str = None
            ) -> Dict[str, Any]:
                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=context,
                    expected_output=expected
                )
                
                results = {}
                for name, metric in [
                    ('faithfulness', self.faithfulness),
                    ('answer_relevancy', self.relevancy),
                    ('context_relevancy', self.context_relevancy),
                    ('context_precision', self.precision),
                    ('context_recall', self.recall)
                ]:
                    try:
                        metric.measure(test_case)
                        results[name] = {
                            'score': metric.score,
                            'reason': metric.reason
                        }
                    except Exception as e:
                        results[name] = {'score': 0.5, 'reason': str(e)}
                
                return results
        
        logger.info("DeepEval library available - using DeepEval metrics")
        return DeepEvalEvaluator()
    
    except ImportError:
        logger.info("DeepEval not installed - using custom LLM judges")
        return LLMAsJudgeEvaluator()


# ============================================================================
# Convenience functions
# ============================================================================

def evaluate_rag_response(
    question: str,
    answer: str,
    context_chunks: List[Dict[str, Any]],
    use_deepeval: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a RAG response.
    
    Args:
        question: User query
        answer: Generated answer
        context_chunks: Retrieved context
        use_deepeval: Try to use DeepEval library
    
    Returns:
        Evaluation results dictionary
    """
    if use_deepeval:
        evaluator = create_deepeval_evaluator()
        if hasattr(evaluator, 'evaluate'):
            context_texts = [c.get('text', c.get('content', '')) for c in context_chunks]
            return evaluator.evaluate(question, answer, context_texts)
    
    evaluator = LLMAsJudgeEvaluator()
    report = evaluator.evaluate(question, answer, context_chunks)
    return report.to_dict()


# Backward compatibility
AgronomistAuditor = LLMAsJudgeEvaluator
RAGASEvaluator = LLMAsJudgeEvaluator


def main():
    """Test the LLM-as-Judge evaluator."""
    evaluator = LLMAsJudgeEvaluator()
    
    # Sample test case
    question = "What is the optimal soil moisture for tomato cultivation?"
    context = [
        {"text": "Tomatoes require soil moisture between 60-80% field capacity. For clay-loam soils, this translates to 25-30% volumetric water content. Irrigation should be applied when soil moisture drops below 60%."}
    ]
    answer = "For tomatoes, maintain soil moisture between 60-80% field capacity. In clay-loam soils, aim for 25-30% volumetric water content. Irrigate when moisture falls below 60%."
    
    print("Running LLM-as-Judge Evaluation...")
    report = evaluator.evaluate(question, answer, context)
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(report.to_json())


if __name__ == "__main__":
    main()
