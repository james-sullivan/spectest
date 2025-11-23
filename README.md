# SpecTest

A CLI tool for testing whether language models comply with their specifications using value tradeoff scenarios from the paper ["Stress-Testing Model Specs"](https://arxiv.org/abs/2501.00000) (Zhang et al., 2025).

## Overview

SpecTest helps you verify that a language model behaves according to your defined specification by:

1. Loading conflict scenarios from a curated dataset
2. Generating responses from your target model
3. Using multiple judge models to evaluate compliance
4. Reporting detailed statistics including inter-rater agreement

## Installation

```bash
pip install spectest
```

### From Source

```bash
git clone https://github.com/james-sullivan/spectest.git
cd spectest
pip install -e .
```

## Quick Start

1. Create a specification file (e.g., `my-spec.txt`):

```text
The model should prioritize user safety over convenience.
The model should refuse harmful requests politely.
The model should acknowledge uncertainty when appropriate.
```

2. Get an [OpenRouter API key](https://openrouter.ai/) and set it as an environment variable:

```bash
export OPENROUTER_KEY=your-api-key-here
```

3. Run the checker:

```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4
```

## Usage

### Required Arguments

- `--spec PATH`: Path to your specification file (plain text)
- `--model MODEL`: Target model identifier from OpenRouter (e.g., `anthropic/claude-sonnet-4`)

### Optional Arguments

- `--api-key KEY`: Your OpenRouter API key (defaults to `$OPENROUTER_KEY` environment variable)
- `--scenarios N`: Number of scenarios to sample and test (default: 50)
- `--cache-dir PATH`: Directory to cache API responses (enables caching)
- `--output PATH`: Directory to save detailed results (defaults to current directory)
- `--verbose`: Enable debug logging

### Example Commands

**Basic usage:**
```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4
```

**Test 100 scenarios with verbose output:**
```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --scenarios 100 \
  --verbose
```

**Test a different model:**
```bash
spectest \
  --spec my-spec.txt \
  --model openai/gpt-4-turbo
```

**Override the API key:**
```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --api-key your-custom-key
```

**Save detailed results to a custom directory:**
```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --output ./my_results/
```

## Output

The tool displays a summary including:

```
Testing Model: anthropic/claude-sonnet-4
Scenarios: 50

Generating responses... [████████████] 50/50
Evaluating compliance... [████████████] 150/150

RESULTS
═══════════════════════════════════
Compliance Rate:           74.0%
Frequent Non-Compliance:   8.0% (4 scenarios)
Judge Agreement (κ):       0.62 (Substantial agreement)

Example Failures:
1. Scenario #12: All judges flagged bias
2. Scenario #23: All judges flagged contradicting principles
3. Scenario #45: All judges flagged safety violation

Done.
```

## Caching

SpecTest can cache all API calls to save time and costs. To enable caching, use the `--cache-dir` option:

```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --cache-dir .cache
```

When caching is enabled:

- **Rerunning tests is fast**: If you run the same test twice, cached responses are reused instantly
- **Save API costs**: No duplicate API calls are made for identical prompts
- **Resume interrupted runs**: If a test is interrupted, rerun it and it will skip completed scenarios

To clear the cache and force fresh API calls, simply delete the cache directory:
```bash
rm -rf .cache
```

**Note**: Without the `--cache-dir` option, caching is disabled and all API calls are made fresh each time.

The cache is powered by [safety-tooling](https://github.com/safety-research/safety-tooling), which provides persistent on-disk caching.

## Detailed Output

SpecTest automatically saves detailed results to a JSONL file. The filename is automatically generated as:
```
results_{model_name}_{timestamp}.jsonl
```

For example: `results_anthropic_claude-sonnet-4_20251101_234530.jsonl`

By default, the file is saved in the current directory. You can specify a different directory with the `--output` option:

```bash
spectest \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --output ./results/
```

This will create a file like `./results/results_anthropic_claude-sonnet-4_20251101_234530.jsonl`

Each line in the JSONL file contains a complete scenario with:
- **scenario_id**: Unique identifier for the scenario
- **scenario_text**: The original scenario text
- **model**: The model being tested
- **model_response**: The full response from the model
- **judgments**: Array of evaluations from each judge model, including:
  - `judge_model`: Name of the judge
  - `reasoning`: Judge's explanation
  - `judgment`: Result ("compliant", "non-compliant", or "ambiguous")

Example JSONL entry:
```json
{
  "scenario_id": 0,
  "scenario_text": "A user asks for help...",
  "model": "anthropic/claude-sonnet-4",
  "model_response": "I'd be happy to help...",
  "judgments": [
    {
      "judge_model": "google/gemini-2.5-flash-lite",
      "reasoning": "The response demonstrates...",
      "judgment": "compliant"
    }
  ]
}
```

The file is written incrementally as each scenario completes, so you can monitor progress even if the run is interrupted.

### Understanding the Metrics

#### Compliance Rate
The percentage of scenarios where a majority of judge models determined the response complied with your specification. Higher is better.

#### Frequent Non-Compliance
The percentage of scenarios where ALL judge models agreed the response was non-compliant or ambiguous. This identifies clear violations of your spec.

#### Judge Agreement (Fleiss' Kappa)
A statistical measure of inter-rater agreement among the judge models:

| Kappa | Interpretation |
|-------|----------------|
| < 0 | Poor agreement |
| 0.00-0.20 | Slight agreement |
| 0.21-0.40 | Fair agreement |
| 0.41-0.60 | Moderate agreement |
| 0.61-0.80 | Substantial agreement |
| 0.81-1.00 | Almost perfect agreement |

Higher kappa values indicate the judges are more consistent in their evaluations, making the results more reliable.

## Judge Models

The tool uses three hardcoded judge models for evaluation:

1. `anthropic/claude-sonnet-4`
2. `openai/o3-mini`
3. `google/gemini-2.0-flash-exp`

All three judges evaluate each response independently, and the results are aggregated to calculate compliance statistics.

## How It Works

1. **Dataset Loading**: Loads value tradeoff scenarios from the "Stress-Testing Model Specs" HuggingFace dataset
2. **Response Generation**: Sends each scenario to your target model via OpenRouter
3. **Judge Evaluation**: Each of the 3 judge models evaluates whether the response complies with your specification
4. **Statistical Analysis**: Calculates compliance rates and Fleiss' Kappa for inter-rater agreement
5. **Results Display**: Shows a formatted summary with key metrics and example failures

## Dataset

The tool uses scenarios from the paper "Stress-Testing Model Specs" by Zhang et al. (2025). These scenarios are designed to test model behavior in situations involving value tradeoffs and conflicts.

The dataset contains scenarios covering various topics including:
- Safety vs. helpfulness
- Privacy vs. transparency
- Individual rights vs. collective benefit
- Short-term vs. long-term considerations

## Troubleshooting

### "Invalid API key or API connection failed"

- Verify your OpenRouter API key is correct
- Check your internet connection
- Ensure you have credits in your OpenRouter account

### "Failed to load dataset"

- Check your internet connection
- The dataset should be available at `jifanz/stress_testing_model_spec` on HuggingFace
- Try running with `--verbose` to see detailed error messages

### High non-compliance rates

- Review your specification for clarity and completeness
- Consider whether your spec contains conflicting requirements
- Check the example failures to understand what went wrong

### Low judge agreement (low kappa)

- Your specification may be ambiguous or unclear
- Consider adding more specific guidelines
- Low kappa can also indicate complex tradeoffs in the scenarios

### Rate limits or timeouts

- Reduce the number of scenarios with `--scenarios 10`
- Check OpenRouter's rate limits for your account tier
- The tool automatically retries failed requests with exponential backoff

## Development

### Running from source

```bash
# Clone the repository
git clone https://github.com/james-sullivan/spectest.git
cd spectest

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/
```

### Project Structure

```
spectest/
├── src/
│   └── spectest/
│       ├── __init__.py      # Package initialization
│       ├── __main__.py      # Module entry point
│       ├── cli.py           # Click CLI interface
│       ├── dataset.py       # HuggingFace dataset loader
│       ├── judge.py         # Judge evaluation logic
│       ├── stats.py         # Statistical calculations
│       └── output.py        # Terminal formatting
├── pyproject.toml           # Package configuration
└── README.md
```

## Citation

If you use this tool in your research, please cite the original paper:

```bibtex
@article{zhang2025stress,
  title={Stress-Testing Model Specs},
  author={Zhang, et al.},
  journal={arXiv preprint arXiv:2501.00000},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/james-sullivan/spectest/issues)
- Check the [documentation](https://github.com/james-sullivan/spectest)

## Roadmap

Future enhancements planned:

- [ ] Custom API endpoints support
- [ ] Configuration files for repeated tests
- [ ] Checkpoint/resume functionality
- [ ] JSON and Markdown output formats
- [ ] Cost estimation before running
- [ ] Library API for programmatic usage
- [ ] Custom judge model selection
- [ ] Parallel/async processing
- [ ] Advanced scenario filtering
