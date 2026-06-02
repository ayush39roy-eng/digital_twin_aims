from fastapi import FastAPI
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from app.routes import router
from app.rag import collection_count

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = collection_count()
    if count == 0:
        print("WARNING: ChromaDB collection is empty. Run scripts/ingest.py first.")
    else:
        print(f"ChromaDB ready — {count} document chunks loaded.")
    yield


app = FastAPI(title="Karpathy Digital Twin", lifespan=lifespan)
app.include_router(router)


@app.get("/")
def index():
    return FileResponse(FRONTEND)
