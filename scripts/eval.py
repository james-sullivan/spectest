"""Main InspectAI task definition for spec compliance testing."""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.solver import generate

from dataset import load_scenarios
from scorer import compliance_scorer


@task
def spec_compliance(
    spec_file: str,
    num_scenarios: int = 5,
    seed: int = 42,
) -> Task:
    """
    Test model compliance with a specification using value tradeoff scenarios.

    This task:
    1. Loads scenarios from the HuggingFace dataset
    2. Generates responses from the target model
    3. Evaluates compliance using multiple judge models
    4. Calculates agreement metrics (Fleiss' Kappa)

    Args:
        spec_file: Path to the specification file to test against
        num_scenarios: Number of scenarios to sample (default: 50)
        seed: Random seed for reproducible sampling (default: 42)

    Usage:
        inspect eval scripts/eval.py \
            --model anthropic/claude-sonnet-4 \
            -T spec_file=my-spec.txt \
            -T num_scenarios=50 \
            -T seed=42

        # View results
        inspect view
    """
    spec_path = Path(spec_file)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_file}")

    spec_text = spec_path.read_text()
    if not spec_text.strip():
        raise ValueError("Specification file is empty")

    return Task(
        dataset=load_scenarios(spec_text, num_scenarios, seed),
        solver=[generate()],
        scorer=compliance_scorer(spec_text),
    )
