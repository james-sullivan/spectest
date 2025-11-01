"""Terminal output formatting."""

import logging
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


class OutputFormatter:
    """Format and display results to terminal."""

    def __init__(self):
        """Initialize the output formatter."""
        self.console = Console()

    def print_header(self, model: str, n_scenarios: int):
        """
        Print the initial header.

        Args:
            model: Target model name
            n_scenarios: Number of scenarios being tested
        """
        self.console.print()
        self.console.print(f"[bold cyan]Testing Model:[/bold cyan] {model}")
        self.console.print(f"[bold cyan]Scenarios:[/bold cyan] {n_scenarios}")
        self.console.print()

    def print_results(
        self,
        compliance_rate: float,
        noncompliance_rate: float,
        failures: List[Dict[str, Any]],
        kappa: float,
        kappa_interpretation: str,
    ):
        """
        Print the final results summary.

        Args:
            compliance_rate: Percentage of compliant scenarios
            noncompliance_rate: Percentage of frequent non-compliance
            failures: List of failure examples
            kappa: Fleiss' Kappa value
            kappa_interpretation: Human-readable interpretation
        """
        self.console.print()
        self.console.print("[bold]RESULTS[/bold]")
        self.console.print("═" * 50)

        # Create results table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row(
            "Compliance Rate:",
            f"[green]{compliance_rate:.1f}%[/green]" if compliance_rate >= 70
            else f"[yellow]{compliance_rate:.1f}%[/yellow]" if compliance_rate >= 50
            else f"[red]{compliance_rate:.1f}%[/red]"
        )

        n_failures = len(failures)
        table.add_row(
            "Frequent Non-Compliance:",
            f"[red]{noncompliance_rate:.1f}%[/red] ({n_failures} scenario{'s' if n_failures != 1 else ''})"
        )

        kappa_color = (
            "green" if kappa >= 0.61
            else "yellow" if kappa >= 0.41
            else "red"
        )
        table.add_row(
            "Judge Agreement (κ):",
            f"[{kappa_color}]{kappa:.2f}[/{kappa_color}] ({kappa_interpretation})"
        )

        self.console.print(table)

        # Print example failures
        if failures:
            self.console.print()
            self.console.print("[bold]Example Failures:[/bold]")
            for i, failure in enumerate(failures[:3], 1):  # Show up to 3 examples
                scenario_id = failure.get("scenario_id", "Unknown")
                judgments = failure.get("judgments", [])

                # Get reasoning summary
                reasons = set()
                for j in judgments:
                    reasoning = j.get("reasoning", "")
                    if reasoning:
                        # Extract key phrases (simple heuristic)
                        if "bias" in reasoning.lower():
                            reasons.add("bias")
                        if "contradict" in reasoning.lower():
                            reasons.add("contradicting principles")
                        if "safety" in reasoning.lower():
                            reasons.add("safety violation")
                        if "harmful" in reasoning.lower():
                            reasons.add("harmful content")

                if not reasons:
                    reasons.add("compliance issue")

                reason_text = ", ".join(sorted(reasons))
                self.console.print(f"{i}. Scenario #{scenario_id}: All judges flagged {reason_text}")

        self.console.print()
        self.console.print("[bold green]Done.[/bold green]")
        self.console.print()

    def print_error(self, message: str):
        """
        Print an error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_warning(self, message: str):
        """
        Print a warning message.

        Args:
            message: Warning message to display
        """
        self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")

    def print_info(self, message: str):
        """
        Print an info message.

        Args:
            message: Info message to display
        """
        self.console.print(f"[cyan]{message}[/cyan]")
