import time
from typing import Annotated

import gradio
import psutil
from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse

from src.api.handlers import InferenceHandler
from src.webapp.gradio_app import iface

server = FastAPI()

@server.get("/")
def redirect():
    return RedirectResponse(url="/app")


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


server = gradio.mount_gradio_app(server, iface, "/app")