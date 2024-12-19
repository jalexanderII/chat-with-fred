from datetime import datetime, timedelta
from enum import Enum
from typing import List, Any, Dict
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from instructor import OpenAISchema
from pydantic import BaseModel
from pydantic import Field

from backend.config.config import DEFAULT_REGION, OBSERVATION_START_DATE


class QueryMetadata(OpenAISchema):
    """
    Structured representation of an economic assets query's key components.
    """
    region: str = Field(
        default=DEFAULT_REGION,
        description="Geographic region for the economic assets, for example 'US' -> 'United States', 'CA' -> 'Canada', 'Eurozone' -> 'European Union'"
    )
    start_date: Optional[str] = Field(
        None,
        description=f"Start date in YYYY-MM-DD format. of {OBSERVATION_START_DATE} is None mentioned"
    )
    end_date: Optional[str] = Field(
        None,
        description="End date in YYYY-MM-DD format"
    )
    economic_concept: str = Field(
        ...,
        description="Main economic concept (e.g., GDP, Inflation, Unemployment)"
    )

    def __str__(self) -> str:
        """String representation of query metadata"""
        date_range = f"from {self.start_date or 'earliest'} to {self.end_date or 'latest'}"
        return f"{self.economic_concept} assets for {self.region} {date_range}"

class SeriesSelection(OpenAISchema):
    """
    Represents the selection and validation of a FRED series for a given query.
    """
    series_id: Optional[str] = Field(
        None,
        description="FRED series ID if found"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for the selection (0-1), >=0.7 is a good threshold, we will decline results below 0.7"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why this series was selected or why no match was found"
    )
    region_match: bool = Field(
        ...,
        description="Whether series matches requested region"
    )

    def is_valid(self) -> bool:
        """Check if the series selection is valid for use"""
        return bool(self.series_id and self.confidence > 0.7 and self.region_match)

class SeriesEnhancement(OpenAISchema):
    """Schema for LLM-generated series enhancements"""
    description: str = Field(
        ...,
        description="Clear, detailed description of what the series measures"
    )
    common_uses: List[str] = Field(
        ...,
        description="List of common use cases for this economic assets"
    )
    related_concepts: List[str] = Field(
        ...,
        description="List of related economic concepts"
    )
    keywords: List[str] = Field(
        ...,
        description="Relevant keywords for searching this series"
    )
    category: str = Field(
        ...,
        description="Primary economic category this series belongs to"
    )
    region: str = Field(
        DEFAULT_REGION,
        description="Geographic region this series pertains to. If you can not tell, default to 'United States'",
    )

class SeriesMapping(BaseModel):
    """
    Maps economic concepts to FRED series with metadata.
    """
    series_id: str
    title: str
    keywords: List[str]
    region: str
    category: str
    description: str = Field("", description="AI-generated description")
    frequency: str
    units: str
    seasonal_adjustment: Optional[str] = None
    common_uses: List[str] = Field(default_factory=list, description="AI-generated common use cases")
    related_concepts: List[str] = Field(default_factory=list, description="AI-generated related economic concepts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional FRED metadata")
    embedding_id: str = Field("", description="Unique identifier for vector storage")

    @classmethod
    def from_fred_series(
            cls,
            series_id: str,
            series_info: pd.Series,
            enhanced_info: SeriesEnhancement,
            keywords: Optional[List[str]] = None
    ) -> "SeriesMapping":
        """
        Create a SeriesMapping instance from FRED series information.

        Args:
            series_id: FRED series identifier
            series_info: Raw series info from FRED API
            enhanced_info: Enhanced information from LLM
            keywords: Optional additional keywords to include

        Returns:
            SeriesMapping instance
        """
        # Convert series_info to dictionary format
        metadata = {
            'title': series_info.get('title', ''),
            'units': series_info.get('units', ''),
            'frequency': series_info.get('frequency', ''),
            'seasonal_adjustment': series_info.get('seasonal_adjustment', ''),
            'notes': series_info.get('notes', ''),
            'last_updated': series_info.get('last_updated', ''),
        }

        # Combine provided keywords with AI-generated ones
        combined_keywords = list(set(
            (keywords or []) + enhanced_info.keywords
        ))

        return cls(
            series_id=series_id,
            embedding_id=f"series_{series_id}",
            title=metadata['title'],
            keywords=combined_keywords,
            region=enhanced_info.region,
            category=enhanced_info.category,
            description=enhanced_info.description,
            frequency=metadata['frequency'],
            units=metadata['units'],
            seasonal_adjustment=metadata.get('seasonal_adjustment'),
            common_uses=enhanced_info.common_uses,
            related_concepts=enhanced_info.related_concepts,
            metadata=metadata
        )

    def to_pinecone_dict(self) -> Dict[str, Any]:
        """
        Convert to a Pinecone-compatible dictionary with flattened metadata.
        Pinecone metadata must be primitive types or lists of strings.
        """
        return {
            "series_id": self.series_id,
            "title": self.title,
            "keywords": self.keywords,
            "region": self.region,
            "category": self.category,
            "description": self.description,
            "frequency": self.frequency,
            "units": self.units,
            "seasonal_adjustment": self.seasonal_adjustment or "",
            "common_uses": self.common_uses,
            "related_concepts": self.related_concepts,
            # Flatten metadata fields we want to preserve
            "metadata_title": self.metadata.get("title", ""),
            "metadata_units": self.metadata.get("units", ""),
            "metadata_frequency": self.metadata.get("frequency", ""),
            "metadata_notes": self.metadata.get("notes", ""),
            "metadata_last_updated": self.metadata.get("last_updated", ""),
        }

    @classmethod
    def from_pinecone_dict(cls, data: Dict[str, Any]) -> "SeriesMapping":
        """
        Create a SeriesMapping instance from Pinecone metadata.
        """
        # Reconstruct metadata dictionary
        metadata = {
            "title": data.get("metadata_title", ""),
            "units": data.get("metadata_units", ""),
            "frequency": data.get("metadata_frequency", ""),
            "notes": data.get("metadata_notes", ""),
            "last_updated": data.get("metadata_last_updated", ""),
        }

        return cls(
            series_id=data["series_id"],
            title=data["title"],
            keywords=data["keywords"],
            region=data["region"],
            category=data["category"],
            description=data["description"],
            frequency=data["frequency"],
            units=data["units"],
            seasonal_adjustment=data["seasonal_adjustment"],
            common_uses=data["common_uses"],
            related_concepts=data["related_concepts"],
            metadata=metadata,
            embedding_id=f"series_{data['series_id']}"
        )

    def matches_query(self, query: str, region: str) -> bool:
        query = query.lower()
        return (
                self.region.lower() == region.lower() and
                any(keyword.lower() in query for keyword in self.keywords)
        )

    def to_embedding_text(self) -> str:
        return self.model_dump_json()


class EconomicAnalysis(OpenAISchema):
    """
    Structured economic assets analysis results.
    """
    latest_value: str = Field(
        ...,
        description="Latest value with proper formatting"
    )
    trend_description: str = Field(
        ...,
        description="Description of recent trend"
    )
    key_observations: List[str] = Field(
        ...,
        description="List of key observations"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in analysis (0-1)"
    )


class QueryInfo(BaseModel):
    """Structure for query metadata"""
    original_query: str
    metadata: dict
    confidence: float

class SeriesInfo(BaseModel):
    """Structure for series information"""
    id: str
    title: str
    units: str
    frequency: str
    last_updated: str
    region_match: bool
    selection_reasoning: str


class SourceInfo(BaseModel):
    """Information about the assets source"""
    series_id: str
    title: str
    observation_start: Optional[str]
    observation_end: Optional[str]
    frequency: str
    seasonal_adjustment: Optional[str]
    units: str
    notes: Optional[str]
    last_updated: str
    source_name: str = "Federal Reserve Economic Data (FRED)"
    source_url: str = "https://fred.stlouisfed.org"

    def format_citation(self) -> str:
        """Format source information as a citation string"""
        date_range = f" ({self.observation_start} to {self.observation_end})" if self.observation_start and self.observation_end else ""

        citation = (
            f"Source: {self.source_name}\n"
            f"Series: {self.title} ({self.series_id}){date_range}\n"
            f"Units: {self.units}"
        )

        if self.seasonal_adjustment:
            citation += f"\nSeasonal Adjustment: {self.seasonal_adjustment}"

        citation += f"\nFrequency: {self.frequency}"
        citation += f"\nLast Updated: {self.last_updated}"
        citation += f"\nRetrieved from: {self.source_url}/series/{self.series_id}"

        return citation

class AnalysisResult(BaseModel):
    """Structure for analysis results"""
    latest_value: str
    trend: str
    key_observations: List[str]
    confidence_score: float

class PlotData(BaseModel):
    """Schema for plot assets including base64 and file information"""
    base64: str = Field(..., description="Base64 encoded plot image")
    filename: str = Field(..., description="Generated filename for the plot")
    path: str = Field(..., description="Full path to saved plot file")

class Visualization(BaseModel):
    """Schema for visualization assets"""
    plot: Optional[PlotData] = Field(None, description="Plot assets including encoded image and file info")
    format: str = Field(default="png", description="Image format")

class ChatbotResponse(BaseModel):
    """Complete response schema including visualization"""
    status: str = Field(..., description="Success or error status")
    message: Optional[str] = Field(None, description="Error message if applicable")
    details: Optional[str] = Field(None, description="Additional error details")
    query_info: Optional[QueryInfo] = Field(None, description="Query metadata")
    series_info: Optional[SeriesInfo] = Field(None, description="Series information")
    analysis: Optional[AnalysisResult] = Field(None, description="Analysis results")
    visualization: Optional[Visualization] = Field(None, description="Visualization assets")
    source_info: Optional[SourceInfo] = Field(None, description="Source information")

    @classmethod
    def create_error(cls, message: str, details: str) -> "ChatbotResponse":
        """Create an error response"""
        return cls(
            status="error",
            message=message,
            details=details
        )

    @classmethod
    def create_success(
        cls,
        query_info: QueryInfo,
        series_info: SeriesInfo,
        analysis: AnalysisResult,
        visualization: Optional[Visualization] = None
    ) -> "ChatbotResponse":
        """Create a success response"""
        return cls(
            status="success",
            query_info=query_info,
            series_info=series_info,
            analysis=analysis,
            visualization=visualization
        )

class QueryRequest(BaseModel):
    query: str


class Period(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    CURRENT = "current"
    EXACT = "exact"


class GetDateRequest(OpenAISchema):
    """
    Represents a single date request, either for start or end date.
    Used to convert natural language date references into structured format.
    """
    period: Period = Field(
        ...,
        description=(
            "The time period unit to use. Choose 'exact' for specific dates, "
            "'current' for current date, or day/week/month/year for relative dates"
        )
    )
    duration: Optional[int] = Field(
        None,
        description=(
            "Number of periods to go back in time. Required when period is "
            "day/week/month/year. Must be >= 1. Not used for 'current' or 'exact' periods"
        ),
        gt=0
    )
    exact_date: Optional[str] = Field(
        None,
        description=(
            "Specific date in MM-DD-YYYY format. Required only when period is 'exact'. "
            "Example: '12-25-2023' for December 25, 2023"
        ),
        pattern=r"^\d{2}-\d{2}-\d{4}$"
    )

class GetDateRequests(OpenAISchema):
    """
    Container for both start and end date requests extracted from natural language query.
    Used to process queries like 'last 10 years' or 'between Jan 1 2020 and Dec 31 2023'
    """
    start_date: GetDateRequest = Field(
        ...,
        description=(
            "Start date specification. For relative queries like 'last X years', "
            "this should use appropriate period and duration. For exact dates, "
            "use period='exact' with exact_date"
        )
    )
    end_date: GetDateRequest = Field(
        ...,
        description=(
            "End date specification. For relative queries, this typically uses "
            "period='current'. For exact date ranges, use period='exact' with exact_date"
        )
    )

    @staticmethod
    def _get_date(period: Period, duration: Optional[int] = None, exact_date: Optional[str] = None) -> str:
        """
        Returns an ISO formatted date string based on the period and duration from current date.

        Args:
            period (Period): The time period unit (day, week, month, year, current, exact)
            duration (int): Number of periods to go back in time, >= 1
            exact_date (str, optional): Exact date in MM-DD-YYYY format

        Returns:
            str: ISO formatted date string (YYYY-MM-DD)
        """
        current_date = datetime.now()

        if period == Period.CURRENT:
            return current_date.date().isoformat()

        if period == Period.EXACT and exact_date:
            # convert exact date in MM-DD-YYYY format to datetime object
            result_date = datetime.strptime(exact_date, "%m-%d-%Y")
            return result_date.date().isoformat()

        if not duration or duration < 0:
            raise ValueError("Duration must be non-negative if period is not 'current' or 'exact'")

        if period == Period.DAY:
            result_date = current_date - timedelta(days=duration)
        elif period == Period.WEEK:
            result_date = current_date - timedelta(weeks=duration)
        elif period == Period.MONTH:
            result_date = current_date - relativedelta(months=duration)
        elif period == Period.YEAR:
            result_date = current_date - relativedelta(years=duration)
        else:
            raise ValueError(f"Invalid period: {period}")

        return result_date.date().isoformat()

    def extract_date_range(self) -> tuple[str, str]:
        """
        Extracts start and end dates from a natural language query using Instructor.
        """
        start_date = None if self.start_date.period == Period.CURRENT else self._get_date(
            period=self.start_date.period,
            duration=self.start_date.duration,
            exact_date=self.start_date.exact_date
        )
        end_date = self._get_date(
            period=self.end_date.period,
            duration=self.end_date.duration,
            exact_date=self.end_date.exact_date
        )
        return start_date, end_date