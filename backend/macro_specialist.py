from typing import Dict, Any

from fredapi import Fred

from backend.analysis import QueryAnalyzer, SeriesAnalyzer, DataAnalyzer
from backend.config.config import logger
from backend.managers import FredManager, DataFormatter, PlotManager
from backend.schemas import ChatbotResponse, AnalysisResult, SeriesInfo, QueryInfo, SourceInfo
from backend.vector_db import VectorDBManager


class MacroSpecialist:
    """Main chatbot class for handling economic assets queries"""

    def __init__(self, fred_client: Fred, vector_db: VectorDBManager):
        self.fred_service = FredManager(fred_client)
        self.query_analyzer = QueryAnalyzer()
        self.series_analyzer = SeriesAnalyzer(fred_client, vector_db)
        self.data_analyzer = DataAnalyzer()
        self.plot_manager = PlotManager()
        self.formatter = DataFormatter()

    async def process_query(self, user_query: str) -> ChatbotResponse:
        """
        Process an economic assets query and return structured response.

        Args:
            user_query: Natural language query about economic assets

        Returns:
            ChatbotResponse containing structured analysis and visualization
        """
        logger.info(f"Processing query: {user_query}")

        try:
            query_metadata = self.query_analyzer.extract_metadata(user_query)
            series_selection = await self.series_analyzer.find_series(
                user_query,
                query_metadata.economic_concept,
                query_metadata.region
            )

            if not series_selection.series_id:
                return ChatbotResponse.create_error(
                    "No matching assets series found",
                    f"{series_selection.reasoning}_{query_metadata.model_dump()}"
                )

            try:
                data, series_info = self.fred_service.get_series_data(
                    series_selection.series_id,
                    query_metadata.start_date,
                    query_metadata.end_date
                )
            except Exception as e:
                return ChatbotResponse.create_error(
                    "Error fetching assets",
                    f"{str(e)}_{query_metadata.model_dump()}"
                )

            latest_value = self.formatter.format_value(
                data.iloc[-1],
                series_info['units']
            )
            analysis = self.data_analyzer.analyze_series(
                user_query,
                data,
                latest_value,
                series_info,
                query_metadata.region
            )

            source_info = SourceInfo(
                series_id=series_selection.series_id,
                title=series_info['title'],
                observation_start=query_metadata.start_date or data.index[0].strftime('%Y-%m-%d'),
                observation_end=query_metadata.end_date or data.index[-1].strftime('%Y-%m-%d'),
                frequency=series_info['frequency'],
                seasonal_adjustment=series_info.get('seasonal_adjustment'),
                units=series_info['units'],
                notes=series_info.get('notes'),
                last_updated=series_info['last_updated']
            )

            visualization = self.plot_manager.create_visualization(
                data,
                series_info['title'],
                series_info['units']
            )

            return ChatbotResponse(
                status="success",
                query_info=QueryInfo(
                    original_query=user_query,
                    metadata=query_metadata.model_dump(),
                    confidence=series_selection.confidence
                ),
                series_info=SeriesInfo(
                    id=series_selection.series_id,
                    title=series_info['title'],
                    units=series_info['units'],
                    frequency=series_info['frequency'],
                    last_updated=series_info['last_updated'],
                    region_match=series_selection.region_match,
                    selection_reasoning=series_selection.reasoning
                ),
                analysis=AnalysisResult(
                    latest_value=analysis.latest_value,
                    trend=analysis.trend_description,
                    key_observations=analysis.key_observations,
                    confidence_score=analysis.confidence_score
                ),
                visualization=visualization,
                source_info=source_info
            )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return ChatbotResponse.create_error(
                "An unexpected error occurred",
                str(e)
            )

    @staticmethod
    def _create_error_response(
            message: str,
            details: str,
            metadata: Any = None
    ) -> Dict[str, Any]:
        """Create a standardized error response"""
        response = {
            'status': 'error',
            'message': message,
            'details': details
        }
        if metadata:
            response['query_metadata'] = metadata.dict()
        return response
