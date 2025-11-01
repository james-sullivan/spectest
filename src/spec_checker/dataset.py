"""Dataset loader for value tradeoff scenarios."""

import logging
import random
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)


class ScenarioDataset:
    """Loader for stress-testing scenarios from HuggingFace."""

    # Dataset from "Stress-Testing Model Specs" paper
    # Note: This dataset name needs to be verified/updated based on actual HuggingFace dataset
    DATASET_NAME = "anthropics/model-spec-stress-tests"

    def __init__(self):
        """Initialize the dataset loader."""
        self.dataset = None

    def load(self) -> bool:
        """
        Load the dataset from HuggingFace.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading dataset: {self.DATASET_NAME}")
            self.dataset = load_dataset(self.DATASET_NAME, split="train")
            logger.info(f"Loaded {len(self.dataset)} scenarios")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def sample_scenarios(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample N random scenarios from the dataset.

        Args:
            n: Number of scenarios to sample

        Returns:
            List of scenario dictionaries containing:
                - text: The scenario text
                - topics: List of topics (optional)
                - value_pairs: List of conflicting values (optional)
                - id: Scenario identifier
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        total = len(self.dataset)
        if n > total:
            logger.warning(f"Requested {n} scenarios but only {total} available")
            n = total

        # Get random indices
        indices = random.sample(range(total), n)

        scenarios = []
        for idx in indices:
            item = self.dataset[idx]
            scenario = {
                "id": idx,
                "text": item.get("scenario", item.get("text", "")),
                "topics": item.get("topics", []),
                "value_pairs": item.get("value_pairs", []),
            }
            scenarios.append(scenario)

        logger.info(f"Sampled {len(scenarios)} scenarios")
        return scenarios
