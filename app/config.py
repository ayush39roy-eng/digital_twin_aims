import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION = "karpathy_docs"
SQLITE_PATH = str(BASE_DIR / "memory.db")

DATA_DIR = BASE_DIR / "data"
