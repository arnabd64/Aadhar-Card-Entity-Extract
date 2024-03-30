FROM python:3.11-slim-bookworm

# environment variables
ENV PORT=8000
ENV WORKERS=2

# build runtime
COPY requirements.txt .
RUN apt update && \
    apt install -f --no-install-suggests -y libopencv-dev && \
    python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r requirements.txt

# copy the source code
COPY . /app
WORKDIR /app

# run the application
EXPOSE ${PORT}
CMD ["python", "main.py"]