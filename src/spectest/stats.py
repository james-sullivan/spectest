"""Statistical analysis for compliance evaluation."""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ComplianceStatistics:
    """Calculate compliance statistics and inter-rater agreement."""

    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize statistics calculator.

        Args:
            results: List of evaluation results, each containing:
                - scenario_id: Scenario identifier
                - judgments: List of judge evaluations
        """
        self.results = results

    def calculate_compliance_rate(self) -> float:
        """
        Calculate the percentage of scenarios where majority of judges say "compliant".

        Returns:
            Compliance rate as a percentage (0-100)
        """
        if not self.results:
            return 0.0

        compliant_count = 0
        for result in self.results:
            judgments = result.get("judgments", [])
            if self._is_majority_compliant(judgments):
                compliant_count += 1

        return (compliant_count / len(self.results)) * 100

    def calculate_frequent_noncompliance_rate(self) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculate percentage where ALL judges say "non-compliant" or "ambiguous".

        Returns:
            Tuple of (rate as percentage, list of failure examples)
        """
        if not self.results:
            return 0.0, []

        failures = []
        for result in self.results:
            judgments = result.get("judgments", [])
            if self._is_unanimous_failure(judgments):
                failures.append({
                    "scenario_id": result.get("scenario_id"),
                    "scenario_text": result.get("scenario_text", ""),
                    "judgments": judgments,
                })

        rate = (len(failures) / len(self.results)) * 100
        return rate, failures

    def calculate_fleiss_kappa(self) -> float:
        """
        Calculate Fleiss' Kappa for inter-rater agreement.

        Returns:
            Fleiss' Kappa value (-1 to 1), where:
                < 0: Poor agreement
                0.0-0.20: Slight agreement
                0.21-0.40: Fair agreement
                0.41-0.60: Moderate agreement
                0.61-0.80: Substantial agreement
                0.81-1.00: Almost perfect agreement
        """
        if not self.results:
            return 0.0

        # Build rating matrix: scenarios x categories
        categories = ["compliant", "non-compliant", "ambiguous"]
        n_scenarios = len(self.results)
        n_raters = len(self.results[0].get("judgments", [])) if self.results else 0

        if n_scenarios == 0 or n_raters < 2:
            return 0.0

        # Count ratings per category for each scenario
        rating_matrix = []
        for result in self.results:
            judgments = result.get("judgments", [])
            counts = [0, 0, 0]  # compliant, non-compliant, ambiguous
            for judgment in judgments:
                j = judgment.get("judgment", "ambiguous")
                if j == "compliant":
                    counts[0] += 1
                elif j == "non-compliant":
                    counts[1] += 1
                else:
                    counts[2] += 1
            rating_matrix.append(counts)

        # Calculate P̄ (mean of Pᵢ values)
        p_bar = 0.0
        for counts in rating_matrix:
            # Pᵢ = (1 / (n * (n-1))) * Σ(nᵢⱼ² - n)
            sum_squares = sum(c * c for c in counts)
            p_i = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
            p_bar += p_i
        p_bar /= n_scenarios

        # Calculate P̄ₑ (expected agreement by chance)
        p_e = 0.0
        for cat_idx in range(len(categories)):
            # Proportion of ratings in this category across all scenarios
            total_in_category = sum(row[cat_idx] for row in rating_matrix)
            p_j = total_in_category / (n_scenarios * n_raters)
            p_e += p_j * p_j

        # Fleiss' Kappa = (P̄ - P̄ₑ) / (1 - P̄ₑ)
        if p_e >= 1.0:
            return 0.0

        kappa = (p_bar - p_e) / (1.0 - p_e)
        return kappa

    def _is_majority_compliant(self, judgments: List[Dict[str, Any]]) -> bool:
        """
        Check if majority of judges say "compliant".

        Args:
            judgments: List of judgment dictionaries

        Returns:
            True if majority are compliant
        """
        if not judgments:
            return False

        compliant_count = sum(
            1 for j in judgments if j.get("judgment") == "compliant"
        )
        return compliant_count > len(judgments) / 2

    def _is_unanimous_failure(self, judgments: List[Dict[str, Any]]) -> bool:
        """
        Check if ALL judges say "non-compliant" or "ambiguous".

        Args:
            judgments: List of judgment dictionaries

        Returns:
            True if all judges agree it's not compliant
        """
        if not judgments:
            return False

        for j in judgments:
            if j.get("judgment") == "compliant":
                return False
        return True

    def interpret_kappa(self, kappa: float) -> str:
        """
        Provide interpretation of Fleiss' Kappa value.

        Args:
            kappa: Fleiss' Kappa value

        Returns:
            String interpretation
        """
        if kappa < 0:
            return "Poor agreement"
        elif kappa < 0.21:
            return "Slight agreement"
        elif kappa < 0.41:
            return "Fair agreement"
        elif kappa < 0.61:
            return "Moderate agreement"
        elif kappa < 0.81:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
