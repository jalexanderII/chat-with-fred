import asyncio
from typing import List

from fredapi import Fred
from pinecone import Pinecone, ServerlessSpec

from backend.config.config import logger, fred, get_embedding
from backend.config.config import make_instructor_call
from backend.config.env import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION, PINECONE_METRIC, \
    PINECONE_CLOUD_PROVIDER, PINECONE_CLOUD_REGION
from backend.schemas import SeriesEnhancement
from backend.schemas import SeriesMapping


class VectorDBConfig:
    def __init__(self, fred_client: Fred):
        self.fred = fred_client
        self._pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.dimension = PINECONE_DIMENSION
        self.metric = PINECONE_METRIC
        self.spec = ServerlessSpec(cloud=PINECONE_CLOUD_PROVIDER, region=PINECONE_CLOUD_REGION)
        self.existing_indexes = [i["name"] for i in self._pinecone_client.list_indexes()]

    async def initialize_index(self):
        try:
            if self.index_name not in self.existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self._pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.spec
                )
                await asyncio.sleep(1)
                await self._seed_vector_db()
            else:
                logger.info(f"Index {self.index_name} already exists, skipping creation")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Index {self.index_name} already exists, continuing...")
            else:
                logger.error(f"Error initializing index: {str(e)}")
                raise

    @property
    def index(self):
        return self._pinecone_client.Index(self.index_name)

    async def series_exists(self, series_id: str) -> bool:
        return bool(self.index.fetch([f"series_{series_id}"]).vectors)


    async def store_series(self, mapping: SeriesMapping):
        text = mapping.to_embedding_text()
        embedding = get_embedding(text)
        self.index.upsert(
            vectors=[(
                mapping.embedding_id,
                embedding,
                mapping.to_pinecone_dict()
            )]
        )

    async def _seed_vector_db(self) -> None:
        """
        Seed the vector database with enhanced versions of core economic series.
        """
        initial_mappings = {
            'US_GDP': {
                'series_id': 'GDP',
                'context': 'Gross Domestic Product, which measures total economic output',
                'keywords': ['gdp', 'economic growth', 'output'],
                'region': 'United States',
                'category': 'GDP'
            },
            'US_INFLATION': {
                'series_id': 'CPIAUCSL',
                'context': 'Consumer Price Index, which measures inflation and price changes',
                'keywords': ['inflation', 'prices', 'cpi', "consumer price index"],
                'region': 'United States',
                'category': 'Inflation'
            },
            'EU_INFLATION': {
                'series_id': "FPCPITOTLZGEUU",
                'context': 'Consumer Price Index, which measures inflation and price changes',
                'keywords': ['inflation', 'prices', 'cpi', 'eurozone', "european union", "eu", "europe", "euro", "consumer price index"],
                'region': 'European Union',
                'category': 'Inflation'
            },
            'US_UNEMPLOYMENT': {
                'series_id': 'UNRATE',
                'context': 'Unemployment Rate, measuring joblessness in the labor force',
                'keywords': ['unemployment', 'jobless', 'jobs'],
                'region': 'United States',
                'category': 'Unemployment'
            },
            'US_INTEREST_RATE': {
                'series_id': 'DFF',
                'context': 'Federal Funds Rate, the key interest rate set by the Federal Reserve',
                'keywords': ['interest', 'rate', 'federal funds rate', 'monetary policy'],
                'region': 'United States',
                'category': 'Interest Rates'
            }
        }

        logger.info("Starting vector database seeding process...")

        for key, mapping in initial_mappings.items():
            series_id = mapping['series_id']

            # Check if series already exists in vector DB
            if await self.series_exists(series_id):
                logger.info(f"Series {series_id} already exists in vector DB, skipping...")
                continue

            try:
                # Get FRED series info
                series_info = self.fred.get_series_info(series_id)

                # Generate enhancement prompt
                enhancement_prompt = (
                    f"Analyze this core FRED economic data series and provide structured information:\n\n"
                    f"Title: {series_info['title']}\n"
                    f"Series ID: {series_id}\n"
                    f"Original Description: {series_info.get('notes', '')}\n"
                    f"Context: {mapping['context']}\n"
                    f"Category: {mapping['category']}\n"
                    f"Units: {series_info.get('units', '')}\n"
                    f"Frequency: {series_info.get('frequency', '')}\n\n"
                    "Provide a comprehensive analysis including:\n"
                    "1. A clear description of what this series measures\n"
                    "2. Common use cases in economic analysis\n"
                    "3. Related economic concepts\n"
                    "4. Relevant search keywords (include existing keywords)\n"
                    "5. The primary economic category"
                )

                # Get enhanced information using LLM
                enhanced_info: SeriesEnhancement = make_instructor_call(
                    enhancement_prompt,
                    "Analyze the FRED series",
                    SeriesEnhancement
                )

                # Create enhanced mapping
                enhanced_mapping = SeriesMapping.from_fred_series(
                    series_id=series_id,
                    series_info=series_info,
                    enhanced_info=enhanced_info,
                    keywords=mapping['keywords']
                )

                # Store in vector DB
                await self.store_series(enhanced_mapping)
                logger.info(f"Successfully enhanced and stored {series_id} in vector DB")

            except Exception as e:
                logger.error(f"Error processing series {series_id}: {str(e)}", exc_info=True)
                continue

        logger.info("Vector database seeding completed")


class VectorDBManager:
    """Manages interactions with Pinecone vector database"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.index = config.index

    async def search_series(self, query: str, top_k: int = 5) -> List[SeriesMapping]:
        query_embedding = get_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        if not results or not results.matches:
            return logger.info(f"No result. Description of index: {self.index.describe_index_stats()}")
        return [SeriesMapping.from_pinecone_dict(match.metadata) for match in results.matches]

    async def store_series(self, mapping: SeriesMapping):
        await self.config.store_series(mapping)

    async def series_exists(self, series_id: str) -> bool:
        return await self.config.series_exists(series_id)



async def get_vector_db() -> VectorDBManager:
    vector_db_config = VectorDBConfig(fred)
    await vector_db_config.initialize_index()
    return VectorDBManager(vector_db_config)
