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
from .pricing import calculate_cost
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
        logging.getLogger("safetytooling").setLevel(logging.ERROR)
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

    # Initialize cost tracking
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    cost_by_model: dict[str, float] = {}  # Track costs per model

    async def generate_with_id(scenario_data):
        """Helper to generate response and keep track of scenario data."""
        response = await judge.generate_response(model, scenario_data["text"])
        if response:
            return {
                "scenario_id": scenario_data["id"],
                "scenario_text": scenario_data["text"],
                "response": response["text"],
                "cost": response["cost"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "scenario_data": scenario_data,
            }
        else:
            logger.warning(f"Failed to generate response for scenario {scenario_data['id']}")
            return None
    
    # Generate all responses in parallel with progress bar
    tasks = [generate_with_id(s) for s in sampled_scenarios]
    results_with_none = []
    for coro in async_tqdm.as_completed(tasks, desc="Generating responses", unit="scenario", file=sys.stderr):
        result = await coro
        results_with_none.append(result)
        # Track costs from response generation
        if result:
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            cost = calculate_cost(model, result["input_tokens"], result["output_tokens"])
            total_cost += cost
            cost_by_model[model] = cost_by_model.get(model, 0.0) + cost

    # Filter out None results
    results = [r for r in results_with_none if r is not None]

    if not results:
        formatter.print_error("Failed to generate any responses")
        sys.exit(1)

    # Phase 3: Judge evaluation in parallel
    formatter.print_info(f"Evaluating compliance with {len(judge.judge_models)} judges...")

    # Create individual tasks for each (scenario, judge) combination
    # This allows progress tracking per individual judgment instead of per scenario
    async def evaluate_single_judge(result, judge_model):
        """Evaluate a single scenario with a single judge."""
        judgment = await judge._get_single_judgment(
            judge_model=judge_model,
            specification=specification,
            scenario=result["scenario_text"],
            model_response=result["response"],
        )
        return {
            "scenario_id": result["scenario_id"],
            "judge_model": judge_model,
            "judgment": judgment,
        }

    # Create flat list of all (scenario, judge) tasks
    judge_tasks = [
        evaluate_single_judge(result, judge_model)
        for result in results
        for judge_model in judge.judge_models
    ]

    expected_judgments = len(judge_tasks)
    successful_judgments = 0

    # Dictionary to accumulate judgments by scenario_id
    scenario_judgments = {r["scenario_id"]: {"judgments": [], "failed_judgments": [], "result": r} for r in results}

    try:
        with tqdm(total=expected_judgments, desc="Evaluating compliance", unit="judgment", file=sys.stderr) as pbar:
            # Process judgments as they complete individually
            for coro in asyncio.as_completed(judge_tasks):
                eval_result = await coro
                scenario_id = eval_result["scenario_id"]
                judge_model = eval_result["judge_model"]
                judgment = eval_result["judgment"]

                # Track costs from judgments
                input_tokens = judgment.get("input_tokens", 0)
                output_tokens = judgment.get("output_tokens", 0)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                cost = calculate_cost(judge_model, input_tokens, output_tokens)
                total_cost += cost
                cost_by_model[judge_model] = cost_by_model.get(judge_model, 0.0) + cost

                # Categorize as success or failure
                if judgment.get("success"):
                    scenario_judgments[scenario_id]["judgments"].append({
                        "judge_model": judge_model,
                        "reasoning": judgment["reasoning"],
                        "judgment": judgment["judgment"],
                    })
                    successful_judgments += 1
                else:
                    scenario_judgments[scenario_id]["failed_judgments"].append({
                        "judge_model": judge_model,
                        "raw_response": judgment.get("raw_response", ""),
                        "error": judgment.get("error", "Unknown error"),
                    })

                # Update progress bar for each individual judgment
                pbar.update(1)

                # Check if all judges for this scenario have completed
                total_judgments_for_scenario = (
                    len(scenario_judgments[scenario_id]["judgments"]) +
                    len(scenario_judgments[scenario_id]["failed_judgments"])
                )

                # Write to JSONL file when all judges for a scenario are done
                if total_judgments_for_scenario == len(judge.judge_models):
                    result_data = scenario_judgments[scenario_id]["result"]
                    scenario_data = result_data["scenario_data"]
                    value_pairs = scenario_data.get("value_pairs", [])
                    value1 = value_pairs[0] if len(value_pairs) > 0 else ""
                    value2 = value_pairs[1] if len(value_pairs) > 1 else ""

                    jsonl_entry = {
                        "scenario_id": scenario_id,
                        "scenario_text": result_data["scenario_text"],
                        "value1": value1,
                        "value2": value2,
                        "nudge_direction": scenario_data.get("nudge_direction", ""),
                        "model": model,
                        "spec_file": spec.name,
                        "model_response": result_data["response"],
                        "judgments": scenario_judgments[scenario_id]["judgments"],
                    }

                    # Include failed_judgments only if there are any
                    if scenario_judgments[scenario_id]["failed_judgments"]:
                        jsonl_entry["failed_judgments"] = scenario_judgments[scenario_id]["failed_judgments"]

                    output_file.write(json.dumps(jsonl_entry) + "\n")
                    output_file.flush()  # Ensure it's written immediately

        # Reset terminal after tqdm progress bars to ensure clean output
        # tqdm uses ANSI codes that can interfere with Rich console
        if sys.stderr.isatty():
            sys.stderr.write('\033[0m')  # Reset all attributes
            sys.stderr.flush()
        print()  # Newline on stdout
        sys.stdout.flush()

        # Convert scenario_judgments back to evaluation_results format
        evaluation_results = []
        for scenario_id, scenario_data in scenario_judgments.items():
            result = scenario_data["result"]
            evaluation_results.append({
                "scenario_id": scenario_id,
                "scenario_text": result["scenario_text"],
                "judgments": scenario_data["judgments"],
                "failed_judgments": scenario_data["failed_judgments"],
            })

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
                total_cost=total_cost,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                cache_enabled=(cache_dir is not None),
                cost_by_model=cost_by_model,
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
