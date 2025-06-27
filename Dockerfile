FROM python:3.12-slim-bookworm

# add debian dependencies
RUN apt update && \
    apt install -y libgl1-mesa-dev libglib2.0-0  && \
    apt clean

# install torchvision
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install requriements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# copy the source code
COPY . /app
WORKDIR /app

# environment variables
ENV MODELS_STORAGE_PATH=/models
EXPOSE ${PORT}

# run the application
RUN mkdir -p ${MODELS_STORAGE_PATH} && python src/core.py
CMD ["python", "main.py"]