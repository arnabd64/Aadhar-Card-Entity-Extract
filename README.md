---
title: Aadhar Card OCR
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 8000
---

# Aadhar Card Entity Extraction

## Overview

Aadhar Card is the official photo ID document in India. This project aims to build an easy to use method to extract the key information from a picture of an Aadhar Card, mainly the Aadhar Number, Name of the Person, Date of Birth and Address. This is made possible using a Text Detection model and a Text Recognition Model.

The __Text Detection__ model is tasked with detecting the four entities on an image of aadhar card and getting the location of bounding boxes for the entities.We have used the YOLOv8 model provided by [Ultralytics](https://docs.ultralytics.com/). The reason to choose YOLOv8 over some other detection model is simply the ease of use throughoout the lifecycle of the model. Ultralytics provides us a low code approach to fine-tune and deploy a YOLOv8 model. Also we want our model to have a low impact on CPU and memory so we opted for the Nano variant of the model.

The __Text Recognition__ model is tasked with identifying text that is contained inside a bounding box. YOLOv8 model provides the bounding box locations which is piped into the text recognition model where it identifies the text in each box and returns the overall output in a JSON object. example:

For the front-side, expect the following output:
```json
{
    "AADHAR_NUMBER": "123456789012",
    "NAME": "John Doe",
    "DATE_OF_BIRTH": "01-01-1997",
}
```

For the back-side, expect the following output:
```json
{
    "ADDRESS": "Some Random Address",
    "AADHAR_NUMBER": "123456789012"
}
```

## How to use

There are a two ways that you can use this model:

1. Using as a __Gradio__ web app.
2. Using as a __FastAPI__ Server.

### Gradio Web App

Download the source code:

```bash
# clone the repo
$ git clone https://github.com/arnabd64/Aadhar-Card-Entity-Extract.git

# change to working directory
$ cd Aadhar-Car-Entity-Extract
```

Then create a virtual environment using `venv` or `conda` which ever you prefer. Please use Python 3.9, 3.10 or 3.11

Then install the dependencies

```bash
$ pip install ultralytics supervision gradio huggingface_hub easyocr
```

Run the webapp:

```bash
$ python app.py
```
__Noote:__: The first run will take some since the models will download followed by loading them into memory.

### FastAPI Server

Download the Source Code:

```bash
# clone the repo
$ git clone https://github.com/arnabd64/Aadhar-Card-Entity-Extract.git

# change to working directory
$ cd Aadhar-Car-Entity-Extract
```

#### Run from Source Code

Create a virtual environment using `venv` or `conda` whichever you prefer. Please use Python 3.9, 3.10 or 3.11

Install dependencies

```bash
$ pip install ultralytics supervision huggingface_hub easyocr fastapi uvicorn python-multipart
```

Run the server

```bash
$ uvicorn "server:server" --host=127.0.0.1 --port=8000 --workers=4
```

#### Using Docker

__Docker__

```bash
# build the image
$ docker build -t server:latest .

# run container
$ docker run -itd --rm --name server -p 8000:8000 server:latest
```

__Docker Compose__

```bash
$ docker compose up -d
```

# Links

+ Fine-Tuned Model: [Huggingface](https://huggingface.co/arnabdhar/YOLOv8-nano-aadhar-card)
+ EasyOCR: [GitHub](https://github.com/JaidedAI/EasyOCR)
+ Notebook used for Fine-Tuning: [Link](./Fine-Tune.ipynb)