"""Dataset loader for value tradeoff scenarios."""

import logging

from datasets import load_dataset
from inspect_ai.dataset import MemoryDataset, Sample

logger = logging.getLogger(__name__)

# Dataset from "Stress-Testing Model Specs" paper (Zhang et al., 2025)
# https://huggingface.co/datasets/jifanz/stress_testing_model_spec
DATASET_NAME = "jifanz/stress_testing_model_spec"
DATASET_CONFIG = "complete"  # Use the complete subset with all 411k scenarios


def load_scenarios(
    spec_text: str,
    num_scenarios: int,
    seed: int = 42,
) -> MemoryDataset:
    """
    Load value tradeoff scenarios from HuggingFace as an InspectAI dataset.

    Args:
        spec_text: The specification text (stored in sample metadata)
        num_scenarios: Number of scenarios to sample
        seed: Random seed for reproducible sampling (default: 42)

    Returns:
        MemoryDataset containing Sample objects with scenario data
    """
    logger.info(f"Loading dataset: {DATASET_NAME} (config: {DATASET_CONFIG})")

    # Load dataset in streaming mode to avoid downloading entire 411k dataset
    hf_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    # Shuffle and sample
    # Using a fixed seed ensures reproducibility across runs for effective caching
    shuffled = hf_dataset.shuffle(seed=seed, buffer_size=10000)
    sampled_items = list(shuffled.take(num_scenarios))

    logger.info(f"Sampled {len(sampled_items)} scenarios")

    # Convert to InspectAI Sample objects
    samples = []
    for idx, item in enumerate(sampled_items):
        # Extract value pairs
        value1 = item.get("value1", "")
        value2 = item.get("value2", "")

        sample = Sample(
            input=item.get("query", ""),
            id=str(idx),
            metadata={
                "specification": spec_text,
                "value1": value1,
                "value2": value2,
                "nudge_direction": item.get("nudge_direction", "neutral"),
            },
        )
        samples.append(sample)

    return MemoryDataset(samples)
