import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List

from backend.macro_specialist import MacroSpecialist
from backend.config.config import fred
from backend.vector_db import get_vector_db


def ensure_output_dir(base_path: str) -> Path:
    """
    Ensure output directory exists and return Path object.

    Args:
        base_path: String path to desired output directory

    Returns:
        Path object pointing to created/existing directory
    """
    output_dir = Path(base_path).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
        # Fall back to current directory if we can't create the specified path
        fallback_dir = Path.cwd() / "output"
        fallback_dir.mkdir(exist_ok=True)
        logging.warning(f"Using fallback directory: {fallback_dir}")
        return fallback_dir


def save_plot(base64_data: str, filename: str, output_dir: Path) -> None:
    """Save a base64 encoded plot to a file"""
    import base64

    try:
        plot_path = output_dir / filename
        with open(plot_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        logging.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error saving plot: {str(e)}")


def create_text_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Convert metadata into formatted lines"""
    query_info = metadata['query_info']
    series_info = metadata['series_info']

    return [
        f"Analyzing economic data for {query_info['metadata']['region']} Series: {series_info['title']}",
        f"Frequency: {series_info['frequency']}",
        f"Last Updated: {series_info['last_updated']}",
        ""  # Empty line for spacing
    ]


def create_text_from_analysis(analysis: Dict[str, Any]) -> List[str]:
    """Convert analysis into formatted lines"""
    lines = [
        "Analysis Results:",
        f"Latest Value: {analysis['latest_value']}",
        "",  # Empty line before trend
        f"Trend: {analysis['trend']}",
        "",  # Empty line before observations
        "Key Observations:"
    ]

    # Add observations with bullet points
    for obs in analysis['key_observations']:
        lines.append(f"• {obs}")

    return lines


async def stream_text(lines: List[str]) -> AsyncGenerator[str, None]:
    """Stream text line by line"""
    for line in lines:
        yield json.dumps({
            "type": "text",
            "content": line
        }) + "\n"
        await asyncio.sleep(0.05)  # Small delay between lines


async def initialize_chatbot() -> MacroSpecialist:
    """Initialize the chatbot with all required components"""
    vector_db = await get_vector_db()
    return MacroSpecialist(fred, vector_db)