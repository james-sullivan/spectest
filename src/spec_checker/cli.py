"""CLI interface for spec-checker."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

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
    required=True,
    type=str,
    help="OpenRouter API key"
)
@click.option(
    "--scenarios",
    default=50,
    type=int,
    help="Number of scenarios to test (default: 50)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging"
)
def main(
    spec: Path,
    model: str,
    api_key: str,
    scenarios: int,
    verbose: bool,
):
    """
    Test if a model complies with its specification using value tradeoff scenarios.

    Example:

        spec-checker --spec my-spec.txt --model anthropic/claude-sonnet-4 --api-key $OPENROUTER_KEY
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    formatter = OutputFormatter()

    try:
        # Read specification
        specification = spec.read_text()
        if not specification.strip():
            formatter.print_error("Specification file is empty")
            sys.exit(1)

        # Initialize components
        judge = ComplianceJudge(api_key=api_key)

        # Validate API key
        formatter.print_info("Validating API key...")
        if not judge.validate_api_key():
            formatter.print_error("Invalid API key or API connection failed")
            sys.exit(1)

        # Load dataset
        formatter.print_info("Loading dataset...")
        dataset = ScenarioDataset()
        if not dataset.load():
            formatter.print_error("Failed to load dataset")
            sys.exit(1)

        # Sample scenarios
        sampled_scenarios = dataset.sample_scenarios(scenarios)
        if not sampled_scenarios:
            formatter.print_error("No scenarios sampled")
            sys.exit(1)

        formatter.print_header(model, len(sampled_scenarios))

        # Phase 2: Generate responses
        results = []
        formatter.print_info("Generating responses...")

        for scenario_data in tqdm(sampled_scenarios, desc="Generating responses", unit="scenario"):
            scenario_text = scenario_data["text"]
            scenario_id = scenario_data["id"]

            response = judge.generate_response(model, scenario_text)

            if response:
                results.append({
                    "scenario_id": scenario_id,
                    "scenario_text": scenario_text,
                    "response": response,
                    "scenario_data": scenario_data,
                })
            else:
                logger.warning(f"Failed to generate response for scenario {scenario_id}")

        if not results:
            formatter.print_error("Failed to generate any responses")
            sys.exit(1)

        # Phase 3: Judge evaluation
        formatter.print_info(f"Evaluating compliance with {len(judge.judge_models)} judges...")

        total_evaluations = len(results) * len(judge.judge_models)
        evaluation_results = []

        with tqdm(total=total_evaluations, desc="Evaluating compliance", unit="judgment") as pbar:
            for result in results:
                judgments = judge.evaluate_compliance(
                    specification=specification,
                    scenario=result["scenario_text"],
                    model_response=result["response"],
                )

                evaluation_results.append({
                    "scenario_id": result["scenario_id"],
                    "scenario_text": result["scenario_text"],
                    "response": result["response"],
                    "judgments": judgments,
                })

                # Update progress bar for each judgment received
                pbar.update(len(judgments))

        if not evaluation_results:
            formatter.print_error("Failed to evaluate any scenarios")
            sys.exit(1)

        # Phase 4: Calculate statistics
        stats = ComplianceStatistics(evaluation_results)
        compliance_rate = stats.calculate_compliance_rate()
        noncompliance_rate, failures = stats.calculate_frequent_noncompliance_rate()
        kappa = stats.calculate_fleiss_kappa()
        kappa_interpretation = stats.interpret_kappa(kappa)

        # Phase 5: Display results
        formatter.print_results(
            compliance_rate=compliance_rate,
            noncompliance_rate=noncompliance_rate,
            failures=failures,
            kappa=kappa,
            kappa_interpretation=kappa_interpretation,
        )

    except KeyboardInterrupt:
        formatter.print_warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        formatter.print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
