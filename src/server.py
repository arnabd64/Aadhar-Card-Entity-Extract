import time
from typing import Annotated

import gradio
from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse

from src.gradio_app import app
from src.handlers import APIHealthHandler, InferenceAPIHandler


server = FastAPI()

@server.get("/")
def redirect():
    return RedirectResponse(url="/app")


@server.post("/inference")
async def inference(handler: Annotated[InferenceAPIHandler, Depends()]):
    return await handler.response()


@server.get("/health")
def health(handler: Annotated[APIHealthHandler, Depends()]):
    return handler.response()


@server.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(1000 * (time.perf_counter() - start_time))
    return response


server = gradio.mount_gradio_app(server, app, "/app")