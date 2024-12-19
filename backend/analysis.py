from typing import List

import pandas as pd
from fredapi import Fred

from backend.config.config import logger, DEFAULT_REGION, make_instructor_call, call_llm
from backend.schemas import QueryMetadata, SeriesMapping, SeriesSelection, EconomicAnalysis, \
    SeriesEnhancement, GetDateRequests
from backend.vector_db import VectorDBManager


class QueryAnalyzer:
    """Analyzes and extracts structured information from user queries"""

    def extract_metadata(self, query: str) -> QueryMetadata:
        """
        Extract structured metadata from a user query.

        Args:
            query: Raw user query string

        Returns:
            QueryMetadata object containing structured query information
        """
        start_date, end_date = self.extract_date_range(query)
        instructions = (
            f"Analyze this economic assets query: '{query}'\n"
            f"Extract the region and main economic concept.\n Make sure to convert region to standard format/proper names. "
            f"For example, 'US' should be converted to 'United States', 'Eurozone' to 'European Union', etc."
            f"If region isn't specified, default to '{DEFAULT_REGION}'."
        )

        try:
            qm = make_instructor_call(instructions, "Extract query metadata", QueryMetadata)
            qm.start_date = start_date
            qm.end_date = end_date
            return qm
        except Exception as e:
            logger.error(f"Error extracting query metadata: {str(e)}")
            # Return default metadata if extraction fails
            return QueryMetadata(
                region=DEFAULT_REGION,
                economic_concept=query,
                start_date=None,
                end_date=None
            )

    @staticmethod
    def extract_date_range(query: str) -> tuple[str, str]:
        """
        Extracts start and end dates from a natural language query using Instructor.

        Args:
            query (str): Natural language query about a time period

        Returns:
            GetDateRequests: Structured date range specification
        """
        instructions = """
        Extract start and end dates from the query. Follow these rules:

        1. For relative periods (e.g., "last 10 years"):
           - Start date: Use appropriate period (day/week/month/year) with duration
           - End date: Use period="current"

        2. For exact date ranges (e.g., "between Oct 12 2020 and Nov 19 2023"):
           - Both dates: Use period="exact" with exact_date in MM-DD-YYYY format

        3. For single point references (e.g., "as of January 2023"):
           - Both dates: Use period="exact" with same exact_date

        4. Always ensure dates are properly formatted (MM-DD-YYYY for exact dates)
        """

        date_requests = make_instructor_call(
            instructions=instructions,
            user_prompt=query,
            response_model=GetDateRequests
        )
        return date_requests.extract_date_range()


class SeriesAnalyzer:
    """Handles FRED series selection and analysis"""

    def __init__(self, fred_client: Fred, vector_db: VectorDBManager):
        self.fred = fred_client
        self.vector_db = vector_db

    async def find_series(self, user_query:str, concept: str, region: str) -> SeriesSelection:
        """
        Find the most appropriate FRED series using vector search first,
        then falling back to traditional search if needed.
        """
        logger.info(f"Searching for series matching concept: {concept}, region: {region}")

        try:
            # First try vector search
            search_query = f"user query: {user_query}, concept: {concept}, region: {region}"
            similar_series = await self.vector_db.search_series(search_query)

            if similar_series:
                selection = await self._analyze_series_mapping_results(user_query, concept, region, similar_series)
                if selection.is_valid():
                    logger.info(f"Found matching series via vector search: {selection.series_id}")
                    return selection

            # Use AI to search FRED if no mapping found
            search_query = self._generate_search_query(user_query, concept, region)
            results = self.fred.search(search_query, limit=500)

            if results is None or results.empty:
                return self._create_no_match_selection(concept, region)

            series_mappings = self._convert_fred_results_to_mappings(results)
            selection = await self._analyze_series_mapping_results(user_query, concept, region, series_mappings)

            # If we found a new series, enhance and store it
            if selection.is_valid():
                await self._enhance_and_store_series(selection.series_id, concept)
                logger.info(f"Stored new series in vector DB: {selection.series_id}")

            return selection

        except Exception as e:
            logger.error(f"Error finding series: {str(e)}", exc_info=True)
            return self._create_no_match_selection(concept, region)

    @staticmethod
    def _convert_fred_results_to_mappings(results: pd.DataFrame) -> List[SeriesMapping]:
        """
        Convert FRED search results DataFrame to list of SeriesMapping objects.

        Args:
            results: DataFrame from FRED search API

        Returns:
            List of SeriesMapping objects with basic information
        """
        mappings = []
        for _, row in results.head().iterrows():
            try:
                # Create a basic SeriesMapping with available information
                mapping = SeriesMapping(
                    series_id=row['id'],
                    title=row['title'],
                    keywords=[],  # Will be enhanced later if selected
                    region='',  # Default, will be enhanced later
                    category=row.get('group_id', 'Unknown'),
                    description=row.get('notes', ''),
                    frequency=row.get('frequency', 'Unknown'),
                    units=row.get('units', 'Unknown'),
                    seasonal_adjustment=row.get('seasonal_adjustment'),
                    metadata={
                        'title': row['title'],
                        'notes': row.get('notes', ''),
                        'frequency': row.get('frequency', 'Unknown'),
                        'units': row.get('units', 'Unknown'),
                        'seasonal_adjustment': row.get('seasonal_adjustment'),
                        'last_updated': row.get('last_updated', 'Unknown'),
                    },
                    embedding_id=f"series_{row['id']}"
                )
                mappings.append(mapping)
            except Exception as e:
                logger.error(f"Error converting series {row.get('id', 'unknown')}: {str(e)}")
                continue

        return mappings

    @staticmethod
    async def _analyze_series_mapping_results(
            user_query: str,
            concept: str,
            region: str,
            similar_series: List[SeriesMapping]
    ) -> SeriesSelection:
        """Analyze vector search results using LLM"""
        prompt = (
            f"Analyze these FRED series for the concept '{concept}' in {region}:\n"
            "\n".join(f"- {s.title} ({s.series_id}): {s.description}"
                      for s in similar_series)
        )

        return make_instructor_call(
            prompt,
            f"Select the best matching series for this user query: {user_query}",
            SeriesSelection
        )

    async def _enhance_and_store_series(self, series_id: str, context_query: str):
        """Enhance series metadata and store in vector DB"""
        if await self.vector_db.series_exists(series_id):
            return

        series_info = self.fred.get_series_info(series_id)

        enhancement_prompt = (
            f"Analyze this FRED economic assets series and provide structured information:\n\n"
            f"Title: {series_info['title']}\n"
            f"Original Description: {series_info.get('notes', '')}\n"
            f"Units: {series_info.get('units', '')}\n"
            f"Frequency: {series_info.get('frequency', '')}\n"
            f"Context Query: {context_query}\n\n"
            "Provide a comprehensive analysis including:\n"
            "1. A clear description of what this series measures\n"
            "2. Common use cases in economic analysis\n"
            "3. Related economic concepts\n"
            "4. Relevant search keywords\n"
            "5. The primary economic category"
        )

        enhanced_info: SeriesEnhancement = make_instructor_call(
            enhancement_prompt,
            "Analyze the FRED series",
            SeriesEnhancement
        )

        mapping = SeriesMapping.from_fred_series(
            series_id=series_id,
            series_info=series_info,
            enhanced_info=enhanced_info,
        )

        await self.vector_db.store_series(mapping)


    @staticmethod
    def _generate_search_query(user_query:str, concept: str, region: str) -> str:
        """Generate optimized FRED search query"""
        prompt = f"""Given:
        Original Query: {user_query}
        Economic Concept: {concept}
        Region: {region}

        Create a search query optimized for the FRED database.
        Consider:
        1. Technical economic terms
        2. Unpack Standard abbreviations (e.g., 'CPI' to 'Consumer Price Index')
        3. Regional identifiers. Use standard names (e.g., 'United States' instead of 'US', "European Union" instead of "Eurozone")
        4. Category terms
        5. Combine alternate phrasings, for example, 'unemployment rate / jobless rate' or "Inflation (CPI)"
        6. Combine various regional identifiers (e.g., 'US / United States /USA' or 'Eurozone / European Union / Europe')

        Return only the optimized search terms, no explanation."""
        return call_llm(prompt).strip()

    @staticmethod
    def _analyze_search_results(
            results: pd.DataFrame,
            concept: str,
            region: str
    ) -> SeriesSelection:
        """Analyze FRED search results to select best match"""
        prompt = (
            f"Select the best FRED series for:\n"
            f"Concept: {concept}\n"
            f"Region: {region}\n"
            f"From these options:\n"
            f"{results[['id', 'title', 'notes']].head().to_string()}"
        )
        return make_instructor_call(prompt, "Select the best FRED series", SeriesSelection)

    @staticmethod
    def _create_no_match_selection(concept: str, region: str) -> SeriesSelection:
        """Create a SeriesSelection for when no match is found"""
        return SeriesSelection(
            series_id=None,
            confidence=0.0,
            reasoning=f"No assets found for {concept} in {region}",
            region_match=False
        )


class DataAnalyzer:
    """Analyzes economic assets and generates insights"""

    @staticmethod
    def analyze_series(
            user_query: str,
            data: pd.Series,
            latest_value: str,
            series_info: dict,
            region: str
    ) -> EconomicAnalysis:
        """
        Generate structured analysis of economic time series assets.

        Args:
            user_query: Original user query
            data: Time series assets
            latest_value: Latest assets value formatted with units
            series_info: Series metadata
            region: Geographic region

        Returns:
            EconomicAnalysis containing structured insights
        """
        try:
            instructions = (
                f"Create a natural response for:\n"
                f"Query: {user_query}\n"
                f"Region: {region}:\n"
                f"Series: {series_info['title']}\n"
                f"Latest Value: {latest_value}\n"
                f"Time Range: {data.index[0]} to {data.index[-1]}\n"
                f"Provide structured analysis including latest value, "
                f"Focus on answering the query directly with relevant context."
                f"trend, and key observations."
            )
            return make_instructor_call(instructions, "Analyze the economic assets", EconomicAnalysis)
        except Exception as e:
            logger.error(f"Error analyzing assets: {str(e)}")
            # Return basic analysis if AI analysis fails
            return EconomicAnalysis(
                latest_value=latest_value,
                trend_description="Analysis unavailable",
                key_observations=["Data available but analysis failed"],
                confidence_score=0.0
            )
