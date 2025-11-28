"""CLI interface for spectest."""

import os
import sys

# Set environment variables BEFORE any other imports to suppress warnings
# This must be done before datasets/transformers are imported
if "--verbose" not in sys.argv:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

import asyncio
import contextlib
from datetime import datetime
import json
import logging
import warnings
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# Suppress stderr during imports in non-verbose mode to hide transformers warnings
if "--verbose" not in sys.argv:
    _devnull = open(os.devnull, 'w')
    with contextlib.redirect_stderr(_devnull):
        from .dataset import ScenarioDataset
        from .judge import ComplianceJudge
    # Don't close _devnull - keeps file descriptor open to avoid issues
else:
    from .dataset import ScenarioDataset
    from .judge import ComplianceJudge

from .output import OutputFormatter
from .stats import ComplianceStatistics

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--spec",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to specification file"
)
@click.option(
    "--model",
    required=True,
    type=str,
    help="Target model identifier (e.g., anthropic/claude-sonnet-4)"
)
@click.option(
    "--api-key",
    default=lambda: os.environ.get("OPENROUTER_KEY", ""),
    type=str,
    help="OpenRouter API key (defaults to $OPENROUTER_KEY environment variable)"
)
@click.option(
    "--scenarios",
    default=50,
    type=int,
    help="Number of scenarios to sample and test (default: 50)"
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to cache API responses (enables caching)"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save detailed results (defaults to current directory)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output (show all status messages and debug logging)"
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for scenario sampling (default: 42 for reproducibility)"
)
def main(
    spec: Path,
    model: str,
    api_key: str,
    scenarios: int,
    cache_dir: Path,
    output: Path,
    verbose: bool,
    seed: int,
):
    """
    Test if a model complies with its specification using value tradeoff scenarios.

    Example:

        spectest --spec my-spec.txt --model anthropic/claude-sonnet-4
    """
    # Suppress Python warnings in normal mode
    if not verbose:
        warnings.filterwarnings("ignore")

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure external library logging levels
    if not verbose:
        # Suppress logs from external libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("safetytooling").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    try:
        asyncio.run(async_main(spec, model, api_key, scenarios, cache_dir, output, verbose, seed))
    except KeyboardInterrupt:
        formatter = OutputFormatter()
        formatter.print_warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        formatter = OutputFormatter()
        formatter.print_error(f"Unexpected error: {e}")
        sys.exit(1)


async def async_main(
    spec: Path,
    model: str,
    api_key: str,
    scenarios: int,
    cache_dir: Path,
    output: Path,
    verbose: bool,
    seed: int,
):
    """Async implementation of the main CLI logic."""
    formatter = OutputFormatter(verbose=verbose)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model.replace("/", "_").replace(":", "_")
    filename = f"results_{safe_model_name}_{timestamp}.jsonl"
    
    # Determine output directory (default to current directory)
    output_dir = output if output else Path.cwd()
    output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full output path
    output_path = output_dir / filename
    
    # Open output file
    output_file = open(output_path, "w", encoding="utf-8")
    formatter.print_info(f"Writing detailed results to: {output_path}", force=True)

    # Validate API key
    if not api_key or not api_key.strip():
        formatter.print_error(
            "OpenRouter API key is required. "
            "Provide it via --api-key or set the OPENROUTER_KEY environment variable."
        )
        sys.exit(1)
    
    # Read specification
    specification = spec.read_text()
    if not specification.strip():
        formatter.print_error("Specification file is empty")
        sys.exit(1)

    # Initialize components
    if cache_dir:
        formatter.print_info(f"Caching enabled at: {cache_dir}")
    judge = ComplianceJudge(api_key=api_key, cache_dir=cache_dir, verbose=verbose)

    # Validate API key
    formatter.print_info("Validating API key...")
    if not await judge.validate_api_key():
        formatter.print_error("Invalid API key or API connection failed")
        sys.exit(1)

    # Load dataset
    formatter.print_info("Loading dataset...")
    dataset = ScenarioDataset()
    if not dataset.load():
        formatter.print_error("Failed to load dataset")
        sys.exit(1)

    # Sample scenarios
    sampled_scenarios = dataset.sample_scenarios(scenarios, seed=seed)
    if not sampled_scenarios:
        formatter.print_error("No scenarios sampled")
        sys.exit(1)

    formatter.print_header(model, len(sampled_scenarios))

    # Phase 2: Generate responses in parallel
    formatter.print_info("Generating responses...")
    
    async def generate_with_id(scenario_data):
        """Helper to generate response and keep track of scenario data."""
        response = await judge.generate_response(model, scenario_data["text"])
        if response:
            return {
                "scenario_id": scenario_data["id"],
                "scenario_text": scenario_data["text"],
                "response": response,
                "scenario_data": scenario_data,
            }
        else:
            logger.warning(f"Failed to generate response for scenario {scenario_data['id']}")
            return None
    
    # Generate all responses in parallel with progress bar
    tasks = [generate_with_id(s) for s in sampled_scenarios]
    results_with_none = []
    for coro in async_tqdm.as_completed(tasks, desc="Generating responses", unit="scenario"):
        result = await coro
        results_with_none.append(result)
    
    # Filter out None results
    results = [r for r in results_with_none if r is not None]

    if not results:
        formatter.print_error("Failed to generate any responses")
        sys.exit(1)

    # Phase 3: Judge evaluation in parallel
    formatter.print_info(f"Evaluating compliance with {len(judge.judge_models)} judges...")

    async def evaluate_with_progress(result):
        """Helper to evaluate and return structured result."""
        judgments, cost = await judge.evaluate_compliance(
            specification=specification,
            scenario=result["scenario_text"],
            model_response=result["response"],
        )
        return {
            "scenario_id": result["scenario_id"],
            "scenario_text": result["scenario_text"],
            "response": result["response"],
            "judgments": judgments,
            "judge_cost": cost,
            "scenario_data": result["scenario_data"],
        }
    
    # Evaluate all scenarios in parallel (each with 3 judges in parallel)
    eval_tasks = [evaluate_with_progress(r) for r in results]
    evaluation_results = []
    expected_judgments = len(results) * len(judge.judge_models)
    successful_judgments = 0
    total_judge_cost = 0.0

    try:
        with tqdm(total=expected_judgments, desc="Evaluating compliance", unit="judgment") as pbar:
            for coro in asyncio.as_completed(eval_tasks):
                eval_result = await coro
                evaluation_results.append(eval_result)
                # Track successful judgments and cost
                num_judgments = len(eval_result["judgments"])
                successful_judgments += num_judgments
                total_judge_cost += eval_result.get("judge_cost", 0.0)
                # Update progress for number of judges that succeeded
                pbar.update(num_judgments)
                
                # Write to JSONL file as each scenario completes
                scenario_data = eval_result["scenario_data"]
                value_pairs = scenario_data.get("value_pairs", [])
                value1 = value_pairs[0] if len(value_pairs) > 0 else ""
                value2 = value_pairs[1] if len(value_pairs) > 1 else ""

                jsonl_entry = {
                    "scenario_id": eval_result["scenario_id"],
                    "scenario_text": eval_result["scenario_text"],
                    "value1": value1,
                    "value2": value2,
                    "nudge_direction": scenario_data.get("nudge_direction", ""),
                    "model": model,
                    "spec_file": spec.name,
                    "model_response": eval_result["response"],
                    "judgments": eval_result["judgments"],
                }
                output_file.write(json.dumps(jsonl_entry) + "\n")
                output_file.flush()  # Ensure it's written immediately

        if not evaluation_results:
            formatter.print_error("Failed to evaluate any scenarios")
            sys.exit(1)

        # Show warning if some judgments failed
        failed_judgments = expected_judgments - successful_judgments
        if failed_judgments > 0:
            formatter.print_warning(
                f"{failed_judgments} of {expected_judgments} judgments failed. "
                f"Statistics calculated from {successful_judgments} successful judgments."
            )

        # Phase 4: Calculate statistics
        try:
            stats = ComplianceStatistics(evaluation_results)
            compliance_rate = stats.calculate_compliance_rate()
            noncompliance_rate, failures = stats.calculate_frequent_noncompliance_rate()
            kappa = stats.calculate_fleiss_kappa(expected_judges=len(judge.judge_models))
            kappa_interpretation = stats.interpret_kappa(kappa)

            # Phase 5: Display results
            formatter.print_results(
                compliance_rate=compliance_rate,
                noncompliance_rate=noncompliance_rate,
                failures=failures,
                kappa=kappa,
                kappa_interpretation=kappa_interpretation,
                judge_cost=total_judge_cost,
            )
        except Exception as e:
            logger.exception("Failed to calculate or display statistics")
            formatter.print_error(f"Failed to calculate statistics: {e}")
            formatter.print_info(
                f"Evaluated {len(evaluation_results)} scenarios with "
                f"{successful_judgments}/{expected_judgments} successful judgments.",
                force=True
            )
    finally:
        # Close output file
        output_file.close()
        formatter.print_info(f"Results written to: {output_path}", force=True)


if __name__ == "__main__":
    main()
