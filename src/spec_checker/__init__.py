"""
Spec Checker: CLI tool for testing model compliance with specifications.

This package provides tools to test whether language models comply with
user-defined specifications using value tradeoff scenarios from the
"Stress-Testing Model Specs" paper.
"""

__version__ = "0.1.0"
__author__ = "Spec Checker Contributors"

from .cli import main
from .dataset import ScenarioDataset
from .judge import ComplianceJudge, JUDGE_MODELS
from .output import OutputFormatter
from .stats import ComplianceStatistics

__all__ = [
    "main",
    "ScenarioDataset",
    "ComplianceJudge",
    "JUDGE_MODELS",
    "OutputFormatter",
    "ComplianceStatistics",
]
