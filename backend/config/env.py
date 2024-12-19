import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'fred-search')
PINECONE_CLOUD_PROVIDER = os.environ.get('PINECONE_CLOUD', 'aws')
PINECONE_CLOUD_REGION = os.environ.get('PINECONE_REGION', 'us-east-1')
PINECONE_DIMENSION = int(os.environ.get('PINECONE_DIMENSION', 1536)) # dimensionality of text-embedding-ada-002
PINECONE_METRIC = os.environ.get('PINECONE_METRIC', 'cosine')
EMBED_MODEL = os.environ.get('EMBED_MODEL', "text-embedding-3-small")
FRED_API_KEY = os.environ.get("FRED_API_KEY")

DATA_FOLDER = Path(__file__).parent.parent.parent / "assets"