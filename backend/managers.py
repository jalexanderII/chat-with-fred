import base64
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from fredapi import Fred

from backend.config.config import DataError, FREDError, OBSERVATION_START_DATE, PLOTS_DIR
from backend.config.config import logger
from backend.schemas import Visualization, PlotData


class FredManager:
    """Handles interactions with the FRED API"""

    def __init__(self, fred_client: Fred):
        self.client = fred_client

    def get_series_data(
            self,
            series_id: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> Tuple[pd.Series, dict]:
        """
        Retrieve time series assets and metadata from FRED.

        Args:
            series_id: FRED series identifier
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            Tuple of (time series assets, series metadata)
        """
        try:
            series_info = self.client.get_series_info(series_id)
            data = self.client.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            if data.empty:
                raise DataError(f"No assets available for series {series_id}")

            return data, series_info.to_dict()

        except Exception as e:
            logger.error(f"Error fetching FRED assets: {str(e)}")
            raise FREDError(f"Failed to retrieve assets for series {series_id}: {str(e)}")


class PlotManager:
    def __init__(self):
        self.output_dir = PLOTS_DIR

    def create_visualization(self, data, title: str, units: str) -> Visualization:
        """Create plot and return Visualization object"""
        try:
            plot_data = self.create_and_encode_plot(data, title, units)
            return Visualization(
                plot=PlotData(**plot_data),
                format="png"
            )
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return Visualization(plot=None, format="png")

    def create_and_encode_plot(self, data, title: str, units: str) -> Dict:
        """Create plot and return plot assets dictionary"""
        try:
            # Set up the plot
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(figsize=(12, 7))

            # Plot assets
            ax.plot(data.index, data.values, linewidth=2)

            # Set labels and title
            ax.set_title(title, pad=20)
            ax.set_xlabel('Date', labelpad=10)
            ax.set_ylabel(units, labelpad=10)

            # Add grid and source attribution
            ax.grid(True, alpha=0.3)
            plt.figtext(
                0.99, 0.01,
                'Source: Federal Reserve Economic Data (FRED)',
                ha='right', va='bottom', fontsize=8, style='italic'
            )

            # Generate unique filename
            plot_filename = f"plot_{uuid.uuid4()}.png"
            plot_path = self.output_dir / plot_filename

            # Save to file
            logger.info(f"Saving plot to: {plot_path}")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)

            # Save to BytesIO for base64 encoding
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            encoded_plot = base64.b64encode(buffer.getvalue()).decode()

            plt.close()

            # Verify file was created
            if not plot_path.exists():
                raise FileNotFoundError(f"Failed to save plot to {plot_path}")

            logger.info(f"Successfully created plot: {plot_filename}")
            return {
                "base64": encoded_plot,
                "filename": plot_filename,
                "path": str(plot_path)
            }

        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            raise


class DataFormatter:
    """Handles formatting of economic assets values"""

    @staticmethod
    def format_value(value: float, units: str) -> str:
        """
        Format numeric value based on its units and magnitude.

        Args:
            value: Numeric value to format
            units: Unit type (e.g., 'Percent', 'Dollars', 'Index')

        Returns:
            Formatted string representation
        """
        try:
            if 'percent' in units.lower():
                return f"{value:.1f}%"
            elif 'dollar' in units.lower():
                if abs(value) >= 1e12:
                    return f"${value/1e12:.1f}T"
                elif abs(value) >= 1e9:
                    return f"${value/1e9:.1f}B"
                elif abs(value) >= 1e6:
                    return f"${value/1e6:.1f}M"
                else:
                    return f"${value:,.2f}"
            elif 'index' in units.lower():
                return f"{value:.1f}"
            else:
                return f"{value:,.2f}"

        except Exception as e:
            logger.error(f"Error formatting value: {str(e)}")
            return str(value)

    @staticmethod
    def format_date_range(
            start_date: Optional[str],
            end_date: Optional[str]
    ) -> str:
        """Format date range for display."""
        if not start_date:
            return OBSERVATION_START_DATE

        end_str = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        return f"{start_date} to {end_str}"