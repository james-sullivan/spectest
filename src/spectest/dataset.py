"""Dataset loader for value tradeoff scenarios."""

import logging
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)


class ScenarioDataset:
    """Loader for stress-testing scenarios from HuggingFace."""

    # Dataset from "Stress-Testing Model Specs" paper (Zhang et al., 2025)
    # https://huggingface.co/datasets/jifanz/stress_testing_model_spec
    DATASET_NAME = "jifanz/stress_testing_model_spec"
    DATASET_CONFIG = "complete"  # Use the complete subset with all 411k scenarios

    def __init__(self):
        """Initialize the dataset loader."""
        self.dataset = None
        self.dataset_size = None

    def load(self) -> bool:
        """
        Load the dataset from HuggingFace using streaming mode.
        This downloads data on-demand instead of downloading the entire dataset.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading dataset: {self.DATASET_NAME} (config: {self.DATASET_CONFIG})")
            self.dataset = load_dataset(
                self.DATASET_NAME, 
                self.DATASET_CONFIG, 
                split="train",
                streaming=True
            )
            logger.info("Dataset loaded in streaming mode")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def sample_scenarios(self, n: int, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Sample N random scenarios from the dataset.

        Args:
            n: Number of scenarios to sample
            seed: Random seed for reproducible sampling (default: 42)

        Returns:
            List of scenario dictionaries containing:
                - text: The scenario text (query)
                - value_pairs: List of conflicting values [value1, value2]
                - nudge_direction: Direction of nudge (value1, value2, or neutral)
                - id: Scenario identifier
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Use shuffle and take for streaming datasets
        # This downloads only the requested number of scenarios
        # Using a fixed seed ensures reproducibility across runs for effective caching
        shuffled_dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        sampled_items = list(shuffled_dataset.take(n))

        scenarios = []
        for idx, item in enumerate(sampled_items):
            # Extract value pairs
            value1 = item.get("value1", "")
            value2 = item.get("value2", "")
            value_pairs = [value1, value2] if value1 and value2 else []
            
            scenario = {
                "id": idx,
                "text": item.get("query", ""),
                "value_pairs": value_pairs,
                "nudge_direction": item.get("nudge_direction", "neutral"),
            }
            scenarios.append(scenario)

        logger.info(f"Sampled {len(scenarios)} scenarios")
        return scenarios
