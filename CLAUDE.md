# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecTest is a CLI tool for testing whether language models comply with their specifications using value tradeoff scenarios from the paper "Stress-Testing Model Specs" (Zhang et al., 2025). The tool generates model responses, evaluates them with multiple judge models, and calculates statistical metrics including Fleiss' Kappa for inter-rater agreement.

## Development Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov
```

### Code Quality
```bash
# Format code
black scripts/

# Lint code
ruff check scripts/
```

### Running the Tool
```bash
# Run evaluation using InspectAI
inspect eval scripts/eval.py \
    --model anthropic/claude-sonnet-4 \
    -T spec_file=my-spec.txt \
    -T num_scenarios=50 \
    -T seed=42

# View results
inspect view

# Extract a specific scenario from results
python scripts/extract.py --file results.jsonl --id 0
```

## Architecture

### Core Components

The project is organized as simple scripts in the `scripts/` directory that leverage the InspectAI framework:

**eval.py** - InspectAI task definition
- Defines the `spec_compliance` task using InspectAI's `@task` decorator
- Loads specification file and creates Task with dataset, solver, and scorer
- Entry point for evaluation: `inspect eval scripts/eval.py`

**scorer.py** - Multi-judge compliance scoring
- Implements InspectAI scorer with multiple judge models
- Judge models: `claude-haiku-4.5`, `o4-mini`, `gemini-2.5-flash` (via OpenRouter)
- All judge evaluations run in parallel using `asyncio.gather()`
- Prompts judges to return structured JSON with "reasoning" and "judgment" fields
- Judgment values: "compliant", "non-compliant", or "ambiguous"
- Uses majority vote to determine final compliance score

**metrics.py** - Custom InspectAI metrics
- `compliance_rate()` - Percentage where majority of judges say compliant
- `fleiss_kappa()` - Inter-rater agreement metric
- `frequent_noncompliance()` - Rate where ALL judges say non-compliant/ambiguous

**dataset.py** - Scenario loading
- Loads from HuggingFace dataset `jifanz/stress_testing_model_spec` (config: "complete")
- Uses streaming mode to avoid downloading entire 411k scenario dataset
- Randomly samples N scenarios with shuffling
- Returns InspectAI MemoryDataset

**stats.py** - Statistical calculations
- Implements Fleiss' Kappa calculation for inter-rater agreement
- Provides interpretation of Kappa values (poor to almost perfect agreement)

**extract.py** - Utility script for extracting scenarios
- Reads JSONL results files from InspectAI
- Extracts and formats specific scenarios by ID
- Standalone CLI tool: `python scripts/extract.py --file results.jsonl --id 0`

**pricing.py** - Cost calculation utility
- Fetches model pricing from OpenRouter API
- Used for estimating evaluation costs

### Key Design Patterns

1. **InspectAI Integration**: Uses InspectAI framework for task orchestration, dataset management, scoring, and metrics

2. **Async/Parallel Execution**: All judge evaluations use asyncio for maximum parallelism and throughput

3. **Streaming Dataset**: Uses HuggingFace streaming to sample scenarios without downloading the full 411k dataset

4. **Statistical Rigor**: Uses Fleiss' Kappa (not Cohen's Kappa) because there are 3+ judge models evaluating the same scenarios

5. **Simple Script Architecture**: No package installation required - scripts can be run directly with InspectAI

## Important Implementation Details

### API Integration
- All LLM calls go through InspectAI's model API (using OpenRouter)
- Judge models are hardcoded in `scripts/scorer.py` as `JUDGE_MODELS` constant
- To change judges, modify the `JUDGE_MODELS` list in `scripts/scorer.py`

### Judge Model Configuration
- Current judges: `openrouter/anthropic/claude-haiku-4.5`, `openrouter/openai/o4-mini`, `openrouter/google/gemini-2.5-flash`
- All three judges evaluate each response in parallel
- Majority vote determines final compliance score (2+ compliant = compliant)

### Output Format
- InspectAI generates results in its standard format
- Results include: scenario, model response, score, explanation, and metadata
- Metadata contains: all judgments with reasoning and decisions
- View results with: `inspect view`

### Parallelism Strategy
- Response generation: Handled by InspectAI solver
- Evaluation: All (N scenarios × 3 judges) evaluations run in parallel via asyncio
- InspectAI manages progress tracking and result aggregation
