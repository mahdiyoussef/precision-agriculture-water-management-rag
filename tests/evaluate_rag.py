"""
RAG Evaluation Script - LLM-as-Judge Framework
Uses multiple LLM judges for comprehensive RAG evaluation.
Supports DeepEval integration when available.
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import logger
from src.main import PrecisionAgricultureRAG
from src.utils.evaluator import LLMAsJudgeEvaluator, EvaluationReport, VerdictType


class LLMJudgeEvaluationRunner:
    """
    Runs RAG evaluation using LLM-as-Judge methodology.
    Uses multiple specialized judges for different aspects.
    """
    
    def __init__(self, dataset_path: str = None):
        """Initialize evaluator."""
        self.dataset_path = dataset_path or str(
            Path(__file__).parent / "irrigation_qa_dataset.json"
        )
        self.rag = None
        self.evaluator = None
        self.results = []
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load test dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def initialize(self):
        """Initialize RAG system and evaluator."""
        print("Initializing RAG system...")
        self.rag = PrecisionAgricultureRAG()
        
        print("Initializing LLM-as-Judge Evaluator...")
        self.evaluator = LLMAsJudgeEvaluator()
        
        print("Systems ready.\n")
    
    def evaluate_single_query(
        self,
        query: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a single query using LLM judges."""
        question = query["question"]
        key_concepts = query.get("key_concepts", [])
        
        if verbose:
            print(f"\n[Q{query['id']}] {question[:55]}...")
        
        # Get RAG response
        start_time = time.time()
        response = self.rag.query(question)
        rag_time = time.time() - start_time
        
        generated_answer = response.get("answer", "")
        sources = response.get("sources", [])
        confidence = response.get("confidence", 0)
        
        # Get retrieved context
        retrieved_chunks = getattr(self.rag, '_last_documents', [])
        
        # Run LLM-as-Judge evaluation
        eval_start = time.time()
        report = self.evaluator.evaluate(question, generated_answer, retrieved_chunks)
        eval_time = time.time() - eval_start
        
        # Calculate concept coverage
        concepts_found = [c for c in key_concepts if c.lower() in generated_answer.lower()]
        concept_coverage = len(concepts_found) / len(key_concepts) if key_concepts else 1.0
        
        result = {
            "id": query["id"],
            "question": question,
            "generated_answer": generated_answer[:500],
            "sources": sources,
            "rag_confidence": confidence,
            "concept_coverage": round(concept_coverage, 2),
            "concepts_found": concepts_found,
            "llm_judge_scores": {
                "faithfulness": report.faithfulness.score if report.faithfulness else None,
                "answer_relevancy": report.answer_relevancy.score if report.answer_relevancy else None,
                "context_relevancy": report.context_relevancy.score if report.context_relevancy else None,
                "agricultural_accuracy": report.agricultural_accuracy.score if report.agricultural_accuracy else None,
                "safety": report.safety_assessment.score if report.safety_assessment else None,
            },
            "overall_score": round(report.overall_score, 3),
            "verdict": report.verdict.value,
            "recommendations": report.recommendations,
            "timings": {
                "rag_seconds": round(rag_time, 2),
                "eval_seconds": round(eval_time, 2)
            }
        }
        
        if verbose:
            scores = result["llm_judge_scores"]
            print(f"    Verdict: {report.verdict.value} | Overall: {report.overall_score:.2f}")
            if scores["faithfulness"]:
                print(f"    Faith: {scores['faithfulness']:.2f} | Rel: {scores['answer_relevancy']:.2f} | Agri: {scores['agricultural_accuracy']:.2f}")
            print(f"    Concepts: {concept_coverage:.0%} | Time: {rag_time + eval_time:.1f}s")
        
        return result
    
    def run_evaluation(
        self,
        num_queries: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run full evaluation."""
        dataset = self.load_dataset()
        
        if num_queries:
            dataset = dataset[:num_queries]
        
        if self.rag is None:
            self.initialize()
        
        print(f"\n{'='*70}")
        print(f"LLM-AS-JUDGE EVALUATION - {len(dataset)} Questions")
        print(f"{'='*70}")
        
        self.results = []
        start_time = time.time()
        
        for query in dataset:
            try:
                result = self.evaluate_single_query(query, verbose)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error on query {query['id']}: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()
                self.results.append({
                    "id": query["id"],
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        
        # Calculate summary
        summary = self.calculate_summary(total_time)
        self.print_summary(summary)
        
        return summary
    
    def calculate_summary(self, total_time: float) -> Dict[str, Any]:
        """Calculate evaluation summary."""
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        # Calculate averages for each metric
        metrics = ['faithfulness', 'answer_relevancy', 'context_relevancy', 
                   'agricultural_accuracy', 'safety']
        
        avg_scores = {}
        for metric in metrics:
            scores = [r["llm_judge_scores"].get(metric) for r in valid_results 
                     if r["llm_judge_scores"].get(metric) is not None]
            if scores:
                avg_scores[metric] = round(sum(scores) / len(scores), 3)
        
        avg_overall = sum(r["overall_score"] for r in valid_results) / len(valid_results)
        avg_concept = sum(r["concept_coverage"] for r in valid_results) / len(valid_results)
        avg_confidence = sum(r["rag_confidence"] for r in valid_results) / len(valid_results)
        
        # Timing
        avg_rag = sum(r["timings"]["rag_seconds"] for r in valid_results) / len(valid_results)
        avg_eval = sum(r["timings"]["eval_seconds"] for r in valid_results) / len(valid_results)
        
        # Verdict distribution
        verdicts = {"PASS": 0, "MARGINAL": 0, "FAIL": 0}
        for r in valid_results:
            v = r["verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1
        
        # Collect all recommendations
        all_recs = {}
        for r in valid_results:
            for rec in r.get("recommendations", []):
                all_recs[rec] = all_recs.get(rec, 0) + 1
        top_recs = sorted(all_recs.items(), key=lambda x: -x[1])[:5]
        
        return {
            "total_queries": len(self.results),
            "successful_queries": len(valid_results),
            "failed_queries": len(self.results) - len(valid_results),
            "verdicts": verdicts,
            "pass_rate": (verdicts["PASS"] + verdicts["MARGINAL"] * 0.5) / len(valid_results),
            "llm_judge_scores": avg_scores,
            "overall_score": round(avg_overall, 3),
            "concept_coverage": round(avg_concept, 3),
            "avg_confidence": round(avg_confidence, 3),
            "timing": {
                "avg_rag_seconds": round(avg_rag, 2),
                "avg_eval_seconds": round(avg_eval, 2),
                "total_seconds": round(total_time, 2)
            },
            "top_recommendations": [r[0] for r in top_recs],
            "timestamp": datetime.now().isoformat()
        }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print("LLM-AS-JUDGE EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Successful: {summary['successful_queries']}")
        
        v = summary['verdicts']
        print(f"Verdicts: PASS={v['PASS']}, MARGINAL={v['MARGINAL']}, FAIL={v['FAIL']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        
        print(f"\nLLM Judge Scores (0-1 scale):")
        for metric, score in summary['llm_judge_scores'].items():
            print(f"  {metric.replace('_', ' ').title():25} {score:.3f}")
        
        print(f"\n  {'Overall Score:':25} {summary['overall_score']:.3f}")
        
        print(f"\nConcept Coverage: {summary['concept_coverage']:.1%}")
        print(f"RAG Confidence: {summary['avg_confidence']:.1%}")
        
        print(f"\nTiming:")
        t = summary['timing']
        print(f"  Avg RAG Response: {t['avg_rag_seconds']:.2f}s")
        print(f"  Avg Evaluation: {t['avg_eval_seconds']:.2f}s")
        print(f"  Total Time: {t['total_seconds']:.1f}s")
        
        if summary.get('top_recommendations'):
            print(f"\nTop Recommendations:")
            for rec in summary['top_recommendations'][:3]:
                print(f"  - {rec[:70]}")
        
        print(f"{'='*70}")
    
    def save_results(self, output_path: str = None):
        """Save detailed results to JSON."""
        output_path = output_path or str(
            Path(__file__).parent / f"llm_judge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        output = {
            "summary": self.calculate_summary(0),
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        return output_path


def main():
    """Run LLM-as-Judge evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-as-Judge RAG Evaluation")
    parser.add_argument("--num", "-n", type=int, default=None, help="Number of queries")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()
    
    runner = LLMJudgeEvaluationRunner(dataset_path=args.dataset)
    summary = runner.run_evaluation(num_queries=args.num, verbose=not args.quiet)
    
    if args.save:
        runner.save_results()
    
    # Exit code
    if summary.get("overall_score", 0) >= 0.6:
        print("\n[EVALUATION PASSED]")
        return 0
    else:
        print("\n[EVALUATION NEEDS IMPROVEMENT]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
