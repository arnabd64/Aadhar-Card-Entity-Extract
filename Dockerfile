# STEP 1: build runtime
FROM python:3.11-slim-bookworm

COPY requirements.txt .
RUN apt update && \
    apt install -f -y libopencv-dev && \
    python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "src.server:server", "--host=0.0.0.0","--port=8000", "--workers=4", "--loop=uvloop", "--http=httptools"]