import logging
from pathlib import Path
from typing import List

import instructor
from fredapi import Fred
from openai import OpenAI

from backend.config.env import FRED_API_KEY, OPENAI_API_KEY, EMBED_MODEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OBSERVATION_START_DATE = "2024-01-01"
DEFAULT_REGION = "United States"

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "output" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize global clients
fred = Fred(api_key=FRED_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
iclient = instructor.from_openai(client)

def call_llm(content: str, temperature: float = 0.0) -> str:
    return client.chat.completions.create(model="gpt-4o-mini", temperature=temperature, messages=[{"role": "user", "content": content}]).choices[0].message.content or ""

def get_embedding(query: str) -> List[float]:
    text = query.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding

def make_instructor_call(instructions, user_prompt, response_model):
    return iclient.messages.create(model="gpt-4o", messages=[{"role": "system", "content": instructions}, {"role": "user", "content": user_prompt}
    ], response_model=response_model)

class FREDError(Exception):
    """Base exception for FRED API-related errors"""
    pass

class DataError(Exception):
    """Base exception for assets processing errors"""
    pass