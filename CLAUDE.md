# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecTest is a CLI tool for testing whether language models comply with their specifications using value tradeoff scenarios from the paper "Stress-Testing Model Specs" (Zhang et al., 2025). The tool generates model responses, evaluates them with multiple judge models, and calculates statistical metrics including Fleiss' Kappa for inter-rater agreement.

## Development Commands

### Installation
```bash
# Install in development mode
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
black src/

# Lint code
ruff check src/
```

### Running the Tool
```bash
# Basic usage
spectest --spec my-spec.txt --model anthropic/claude-sonnet-4

# With caching enabled
spectest --spec my-spec.txt --model anthropic/claude-sonnet-4 --cache-dir .cache

# Extract a specific scenario from results
spectest-extract --file results.jsonl --id 0
```

## Architecture

### Core Components

The application follows a modular architecture with clear separation of concerns:

**cli.py** - Entry point and orchestration
- Implements async/await pattern for parallel API requests
- Uses Click for CLI interface
- Coordinates three main phases:
  1. Generate responses from target model (parallelized across scenarios)
  2. Evaluate responses with judge models (parallelized across both scenarios and judges)
  3. Calculate statistics and display results
- Writes incremental JSONL output as each scenario completes

**judge.py** - API interaction and evaluation logic
- Uses `safetytooling.apis.InferenceAPI` for all LLM API calls via OpenRouter
- Supports optional disk-based caching via `cache_dir` parameter (no caching if omitted)
- Hardcoded judge models: `anthropic/claude-sonnet-4`, `openai/o3-mini`, `google/gemini-2.0-flash-exp`
- All judge evaluations run in parallel using `asyncio.gather()`
- Prompts judges to return structured JSON with "reasoning" and "judgment" fields
- Judgment values: "compliant", "non-compliant", or "ambiguous"

**dataset.py** - Scenario loading
- Loads from HuggingFace dataset `jifanz/stress_testing_model_spec` (config: "complete")
- Uses streaming mode to avoid downloading entire 411k scenario dataset
- Randomly samples N scenarios with shuffling

**stats.py** - Statistical analysis
- Calculates compliance rate (majority of judges say "compliant")
- Calculates frequent non-compliance rate (ALL judges say "non-compliant" or "ambiguous")
- Implements Fleiss' Kappa for inter-rater agreement among judge models
- Provides interpretation of Kappa values (poor to almost perfect agreement)

**output.py** - Terminal formatting
- Uses Rich library for formatted console output
- Displays progress bars, results summary, and example failures

**extract.py** - Utility for extracting individual scenarios
- Reads JSONL results files
- Extracts and formats specific scenarios by ID
- Outputs human-readable text files with value tradeoffs, responses, and all judgments

### Key Design Patterns

1. **Async/Parallel Execution**: All API calls (response generation and judging) use asyncio for maximum parallelism and throughput. The tool can process multiple scenarios and multiple judges concurrently.

2. **Incremental Output**: Results are written to JSONL file as each scenario completes, allowing for interruption and partial results inspection.

3. **Optional Caching**: The `safetytooling` library provides disk-based caching when `--cache-dir` is specified. Without it, all requests are made fresh each time.

4. **Streaming Dataset**: Uses HuggingFace streaming to sample scenarios without downloading the full 411k dataset.

5. **Statistical Rigor**: Uses Fleiss' Kappa (not Cohen's Kappa) because there are 3+ judge models evaluating the same scenarios.

## Important Implementation Details

### API Integration
- All LLM calls go through OpenRouter API (https://openrouter.ai/api/v1)
- Uses `safety-tooling` library (from safety-research GitHub) which provides caching and rate limiting
- API key sourced from `--api-key` flag or `$OPENROUTER_KEY` environment variable

### Judge Model Configuration
- Judge models are hardcoded in `judge.py` as `JUDGE_MODELS` constant
- Currently only `anthropic/claude-sonnet-4` is active (others commented out)
- To change judges, modify the `JUDGE_MODELS` list in `src/spectest/judge.py`

### Output Format
- JSONL files include: scenario_id, scenario_text, value1, value2, nudge_direction, model, spec_file, model_response, judgments
- Filename format: `results_{model}_{timestamp}.jsonl`
- Each judgment includes: judge_model, reasoning, judgment

### Parallelism Strategy
- Response generation: All N scenarios run in parallel
- Evaluation: All (N scenarios × 3 judges) evaluations run in parallel
- Progress bars update as individual operations complete
