import time
from typing import Annotated

from fastapi import FastAPI, Depends
import psutil

from src.handlers import InferenceHandler

server = FastAPI(on_startup=[InferenceHandler.download_assets])


@server.get("/")
async def root():
    return {"message": "Welcome to the Inference API!"}


@server.post("/inference")
async def inference(handler: Annotated[InferenceHandler, Depends()]):
    return await handler.build_response()


@server.get("/health")
def health():
    return {
        "status": "UP",
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent
    }
    


@server.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(1000 * (time.perf_counter() - start_time))
    return response
