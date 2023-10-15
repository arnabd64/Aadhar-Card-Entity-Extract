FROM python:3.11-slim-bookworm

COPY . /app
WORKDIR /app


RUN apt update && \
    apt install -f -y libopencv-dev && \
    pip3 install --no-cache-dir --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --quiet huggingface_hub ultralytics supervision easyocr fastapi uvicorn python-multipart && \
    python3 pipelines.py

EXPOSE 8000

CMD ["uvicorn", "server:server", "--host=0.0.0.0", "--workers=4"]