import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config.config import PLOTS_DIR
from backend.macro_specialist import MacroSpecialist
from backend.schemas import ChatbotResponse, QueryRequest
from backend.config.utils import stream_text, create_text_from_analysis, create_text_from_metadata, \
    initialize_chatbot

chatbot: Optional[MacroSpecialist] = None

async def startup_event() -> None:
    global chatbot
    chatbot = await initialize_chatbot()


async def shutdown_event() -> None:
    pass


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    This context manager is used to run startup and shutdown events
    """
    try:
        await startup_event()
        yield
    finally:
        await shutdown_event()

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static files directory with absolute path
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

# Add logging to track file operations
@app.get("/plots/{filename}")
async def get_plot(filename: str):
    file_path = PLOTS_DIR / filename
    print(f"Looking for plot at: {file_path}")  # Debug print
    if not file_path.exists():
        print(f"File not found: {file_path}")  # Debug print
        raise HTTPException(status_code=404, detail=f"Plot not found: {filename}")
    return FileResponse(file_path)

@app.post("/api/chat/stream")
async def stream_chat(request: QueryRequest):
    global chatbot
    if chatbot is None:
        chatbot = await initialize_chatbot()

    try:
        response = await chatbot.process_query(request.query)
        return StreamingResponse(
            stream_response(response),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(response: ChatbotResponse) -> AsyncGenerator[str, None]:
    """Stream the ChatbotResponse with proper formatting"""
    if response.status != "success":
        yield json.dumps({
            "type": "error",
            "content": {
                "message": response.message,
                "details": response.details
            }
        }) + "\n"
        return

    # Stream metadata lines
    metadata_lines = create_text_from_metadata({
        "query_info": response.query_info.model_dump(),
        "series_info": response.series_info.model_dump()
    })

    async for chunk in stream_text(metadata_lines):
        yield chunk

    await asyncio.sleep(0.2)

    # Stream analysis lines
    analysis_lines = create_text_from_analysis(response.analysis.model_dump())
    async for chunk in stream_text(analysis_lines):
        yield chunk

    # If visualization exists, send the plot URL
    if response.visualization and response.visualization.plot:
        await asyncio.sleep(0.2)
        yield json.dumps({
            "type": "visualization",
            "content": {
                "plot_url": f"/plots/{response.visualization.plot.filename}",
                "format": response.visualization.format
            }
        }) + "\n"

    # Stream source information
    if response.source_info:
        await asyncio.sleep(0.2)
        yield json.dumps({
            "type": "source",
            "content": response.source_info.format_citation()
        }) + "\n"




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", reload=True)