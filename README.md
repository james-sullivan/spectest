# Spec Checker

A CLI tool for testing whether language models comply with their specifications using value tradeoff scenarios from the paper ["Stress-Testing Model Specs"](https://arxiv.org/abs/2501.00000) (Zhang et al., 2025).

## Overview

Spec Checker helps you verify that a language model behaves according to your defined specification by:

1. Loading conflict scenarios from a curated dataset
2. Generating responses from your target model
3. Using multiple judge models to evaluate compliance
4. Reporting detailed statistics including inter-rater agreement

## Installation

```bash
pip install spec-checker
```

### From Source

```bash
git clone https://github.com/anthropics/spec-checker.git
cd spec-checker
pip install -e .
```

## Quick Start

1. Create a specification file (e.g., `my-spec.txt`):

```text
The model should prioritize user safety over convenience.
The model should refuse harmful requests politely.
The model should acknowledge uncertainty when appropriate.
```

2. Get an [OpenRouter API key](https://openrouter.ai/)

3. Run the checker:

```bash
spec-checker \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --api-key $OPENROUTER_KEY
```

## Usage

### Required Arguments

- `--spec PATH`: Path to your specification file (plain text)
- `--model MODEL`: Target model identifier from OpenRouter (e.g., `anthropic/claude-sonnet-4`)
- `--api-key KEY`: Your OpenRouter API key

### Optional Arguments

- `--scenarios N`: Number of scenarios to test (default: 50)
- `--verbose`: Enable debug logging

### Example Commands

**Basic usage:**
```bash
spec-checker \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --api-key $OPENROUTER_KEY
```

**Test 100 scenarios with verbose output:**
```bash
spec-checker \
  --spec my-spec.txt \
  --model anthropic/claude-sonnet-4 \
  --api-key $OPENROUTER_KEY \
  --scenarios 100 \
  --verbose
```

**Test a different model:**
```bash
spec-checker \
  --spec my-spec.txt \
  --model openai/gpt-4-turbo \
  --api-key $OPENROUTER_KEY
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
- The dataset should be available at `anthropics/model-spec-stress-tests` on HuggingFace
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
git clone https://github.com/anthropics/spec-checker.git
cd spec-checker

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
spec-checker/
├── src/
│   └── spec_checker/
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
- Open an issue on [GitHub](https://github.com/anthropics/spec-checker/issues)
- Check the [documentation](https://github.com/anthropics/spec-checker)

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
