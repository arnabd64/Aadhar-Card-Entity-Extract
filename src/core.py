import os

import cv2
import numpy as np
import torch
from easyocr import Reader
from fastapi import HTTPException, status
from huggingface_hub import hf_hub_download
from supervision import Detections
from ultralytics import YOLO
from typing import Tuple, Dict

# directory to storage models
MODEL_STORAGE = os.getenv('MODELS_STORAGE_PATH', os.path.join(os.getcwd(), "models"))

# Load YOLOv8
yolo_args = {
    "model": hf_hub_download(
        repo_id = "arnabdhar/YOLOv8-nano-aadhar-card",
        filename = "model.pt",
        local_dir = MODEL_STORAGE
    ),
    "task": "detect"
}
YOLOv8 = YOLO(**yolo_args)

# Load OCR
easyocr_args = {
    "lang_list": ["en"],
    "gpu": False,
    "detector": False,
    "recognizer": True,
    "verbose": False,
    "model_storage_directory": MODEL_STORAGE
}
EASYOCR = Reader(**easyocr_args)


def decode(image_bytes: bytes) -> np.ndarray:
    """
    Decode the image from buffer
    to a numpy array using OpenCV

    Parameters:
        - `image_bytes`: The image buffer

    Returns:
        - `ndarray`: The decoded image
    """
    buffer = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(buffer, cv2.IMREAD_ANYCOLOR)
    return cv_image


def text_detection(image: np.ndarray, model: YOLO = YOLOv8, confidence: float = 0.6):
    """
    Detect text from the image

    Parameters:
        - `image`: The image to detect text from.
        - `model`: Instance of YOLO model.

    Returns:
        - `list`: The detected text
    """
    detections = model.predict(
        image,
        device = torch.device("cpu"),
        conf = confidence,
        verbose = False
    )
    return Detections.from_ultralytics(detections[0])


def validate_inference(detections: Detections):
    """
    Perform the following validations:
    - Check if any entities are detected
    - Check if Aadhar Number is detected
    """
    if len(detections.class_id) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No entities found on the image")
    
    if 0 not in detections.class_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No Aadhar Number Detected")


def postprocess(detections: Detections):
    """
    Postprocess the detections to extract the
    bounding boxes and labels.

    Parameters:
        - `detections`: an instance of Detections

    Returns:
        - `Tuple[np.ndarray, np.ndarray]`: Bounding boxes and labels
    """
    boxes = detections.xyxy.astype(np.uint16)
    labels = detections.class_id.astype(np.uint8)
    return boxes, labels


def text_box(image: np.ndarray, box: Tuple):
     """
     Extracts the text box from the `image` with
     the provided corner points `box`.

     Parameters:
        - `image`: OpenCV Image
        - `box`: Tuple of corner points
    
     Returns:
        - `np.ndarray`: The cropped image of text box
     """
     return image[box[1]:box[3], box[0]:box[2]]


def image_to_text(image: np.ndarray, boxes: Detections, ocr: Reader = EASYOCR):
    """
    Extract the text from the detected bounding boxes
    using EasyOCR.

    Parameters:
        - `image`: Main Image
        - `boxes`: Bounding boxes from YOLOv8
        - `ocr`: Instance of EasyOCR

    Returns:
        - `List[str]`: List of extracted texts
    """
    return [ocr.recognize(text_box(image, box), detail=0)[0] for box in boxes]


def pipeline(image: np.ndarray, confidence: float = 0.6, detector: YOLO = YOLOv8, ocr: Reader = EASYOCR) -> Dict[str, str]:
    """
     Combines all the functions to create a single pipeline
     for Detecting Text fields and then extracting the text.

     Parameters:
        - `image`: The image to process
        - `confidence`: Confidence threshold for Text Detection

    Returns:
        - `Dict[str, str]`: Extracted Text Fields
    """
    # Detect Text
    detections = text_detection(image, detector, confidence)

    # Validate Detection Inference
    validate_inference(detections)
    
    # Postprocess Detections
    boxes, labels = postprocess(detections)

    # Extract Text
    texts = image_to_text(image, boxes, ocr)

    # build the dictionary
    return {label: text for label, text in zip(labels, texts)}
