from dotenv import load_dotenv
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Load environment variables from .env file
load_dotenv()


class Config:
    UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")

    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "articles")

    # Embeddings and fetched articles storage
    # EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "/app/data/embeddings")
    FETCHED_ARTICLES_DIR = os.getenv(
        "FETCHED_ARTICLES_DIR", "/app/data/fetched_articles"
    )

    # LLM configuration
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

    # Ollam configuration
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
    MAX_MODEL_SIZE: int = 4 * 1024**3  # e.g. "7B", "13B", "70B"

    # sqlite configuration
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "sqlite:////app/data/papers.db")


settings = Config()
