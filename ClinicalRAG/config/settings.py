from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage environment variables"""

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

    HF_API_KEY = os.getenv("HF_API_KEY", "")

settings = Config()