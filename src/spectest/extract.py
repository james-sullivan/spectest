"""CLI for extracting events from JSONL result files."""

from datetime import datetime
import json
import logging
import re
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file containing results"
)
@click.option(
    "--id",
    "event_id",
    required=True,
    type=int,
    help="Scenario ID to extract"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: event_{id}_{timestamp}.txt in current directory)"
)
def main(file: Path, event_id: int, output: Path):
    """
    Extract a specific event from a JSONL results file by ID and save as a text file.

    Example:

        spectest-extract --file results.jsonl --id 0
        spectest-extract --file results.jsonl --id 0 --output my_event.txt
    """
    try:
        # Read the JSONL file and find the matching event
        event_found = None
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        if event.get("scenario_id") == event_id:
                            event_found = event
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line: {e}")
                        continue

        if event_found is None:
            click.echo(f"Error: No event found with ID {event_id}", err=True)
            return

        # Format the output
        output_text = format_event(event_found)

        # Determine output file path
        if output is None:
            # Extract timestamp from the JSONL filename if it matches the pattern
            # Expected pattern: results_{model}_{timestamp}.jsonl
            filename_pattern = r'_(\d{8}_\d{6})\.jsonl$'
            match = re.search(filename_pattern, str(file))

            if match:
                timestamp = match.group(1)
            else:
                # Fallback to current timestamp if pattern not found
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output = Path(f"event_{event_id}_{timestamp}.txt")

        # Write to file
        output.write_text(output_text, encoding='utf-8')
        click.echo(f"Event {event_id} written to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def format_event(event: dict) -> str:
    """Format an event as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"SCENARIO ID: {event.get('scenario_id')}")
    lines.append("=" * 80)
    lines.append("")

    # Add value tradeoff information
    value1 = event.get('value1', '')
    value2 = event.get('value2', '')
    nudge_direction = event.get('nudge_direction', '')

    if value1 or value2 or nudge_direction:
        lines.append("VALUE TRADEOFF:")
        lines.append("-" * 80)
        if value1:
            lines.append(f"Value 1: {value1}")
        if value2:
            lines.append(f"Value 2: {value2}")
        if nudge_direction:
            lines.append(f"Nudge Direction: {nudge_direction}")
        lines.append("")

    lines.append("SCENARIO TEXT:")
    lines.append("-" * 80)
    lines.append(event.get('scenario_text', ''))
    lines.append("")

    lines.append("MODEL:")
    lines.append("-" * 80)
    lines.append(event.get('model', ''))
    lines.append("")

    lines.append("MODEL RESPONSE:")
    lines.append("-" * 80)
    lines.append(event.get('model_response', ''))
    lines.append("")

    judgments = event.get('judgments', [])
    if judgments:
        lines.append("JUDGMENTS:")
        lines.append("-" * 80)
        for i, judgment in enumerate(judgments, 1):
            lines.append(f"\nJudgment {i}:")
            lines.append(f"  Judge Model: {judgment.get('judge_model', 'N/A')}")
            lines.append(f"  Judgment: {judgment.get('judgment', 'N/A')}")
            lines.append("  Reasoning:")
            reasoning = judgment.get('reasoning', 'N/A')
            # Indent reasoning
            for reasoning_line in reasoning.split('\n'):
                lines.append(f"    {reasoning_line}")

    lines.append("")
    lines.append("=" * 80)

    return '\n'.join(lines)


if __name__ == "__main__":
    main()
