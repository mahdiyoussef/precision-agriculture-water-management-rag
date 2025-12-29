# RAG Evaluation Test Suite

This directory contains the test dataset and LLM-as-Judge evaluation for the Precision Agriculture RAG system.

## Files

- `irrigation_qa_dataset.json` - 50 irrigation engineering Q&A pairs
- `evaluate_rag.py` - LLM-as-Judge evaluation script

## LLM-as-Judge Methodology

Uses 5 specialized LLM judges:

| Judge | Description |
|-------|-------------|
| **Faithfulness** | Are claims grounded in context? |
| **Answer Relevancy** | Does answer address the question? |
| **Context Relevancy** | Is retrieved context useful? |
| **Agricultural Accuracy** | Domain-specific correctness |
| **Safety** | Is advice safe to follow? |

## Running Evaluation

```bash
# Quick test (3 questions)
python tests/evaluate_rag.py --num 3

# Full evaluation (all 50)
python tests/evaluate_rag.py

# Save results to JSON
python tests/evaluate_rag.py --num 10 --save
```

## Output Example

```
LLM Judge Scores (0-1 scale):
  Faithfulness              0.600
  Answer Relevancy          0.667
  Context Relevancy         0.633
  Agricultural Accuracy     0.667
  Safety                    0.500
  Overall Score:            0.620
```

## DeepEval Integration

Install DeepEval for additional metrics:
```bash
pip install deepeval ragas
```

## Output Format

```json
{
  "summary": {
    "pass_rate": 0.85,
    "scores": {"faithfulness": 4.2, "precision": 4.0, "safety": 4.5},
    "concept_coverage": 0.78
  },
  "results": [...]
}
```
