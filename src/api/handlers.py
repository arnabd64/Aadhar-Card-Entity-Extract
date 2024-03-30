import os
from typing import Optional

import cv2
import easyocr
import numpy as np
import torch
from fastapi import HTTPException, UploadFile, status
from huggingface_hub import hf_hub_download
from supervision import Detections
from ultralytics import YOLO

MODELS_PATH = os.path.join(os.getcwd(), "models")

def download_assets(model_path = MODELS_PATH):
    """
    Download the assets required for the inference
    """
    # create models directory
    os.makedirs(model_path, exist_ok=True)

    # download the YOLO model
    hf_hub_download(
        repo_id = "arnabdhar/YOLOv8-nano-aadhar-card",
        filename = "model.pt",
        local_dir = model_path
    )

    # download the EasyOCR model
    params = {
        "lang_list": ["en"],
        "gpu": False,
        "detector": False,
        "recognizer": True,
        "verbose": True,
        "model_storage_directory": model_path
    }
    _ = easyocr.Reader(**params)

    return None

class InferenceHandler:

    __slots__ = ("image", "yolo", "reader")

    def __init__(self, image: UploadFile):
        self.image = image
        self.yolo = YOLO(os.path.join(MODELS_PATH, "model.pt"), task="detect")
        self.reader = easyocr.Reader(["en"], gpu=False, detector=False, recognizer=True, verbose=False, model_storage_directory=MODELS_PATH)


    async def decode(self) -> np.ndarray:
        """
        Decode the image from buffer
        to a numpy array using OpenCV

        Returns:
            - `ndarray`: The decoded image
        """
        buffer = await self.image.read()
        buffer = np.frombuffer(buffer, np.uint8)
        cv_image = cv2.imdecode(buffer, cv2.IMREAD_ANYCOLOR)
        return cv_image
    

    def text_detection(self, image:np.ndarray):
        """
        Predict the Text bounding boxes in the uploaded
        image using `self.yolo`.

        Parameters:
            - `image`: OpenCV image

        Returns:
            - `Detections`: an instance of Detections
        """
        detections = self.yolo.predict(
            image,
            device = torch.device("cpu"),
            conf = float(os.getenv("YOLO_CONFIDENCE", "0.6")),\
            verbose = False
        )
        return Detections.from_ultralytics(detections[0])
    

    def validate_inference(self, detections: Detections):
        """
        Perform the following validations:
        - Check if any entities are detected
        - Check if Aadhar Number is detected
        """
        if len(detections.class_id) == 0:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No entities found on the image")
        
        elif 0 not in detections.class_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No Aadhar Number Detected")
        
        else:
            pass


    def postprocess(self, detections: Detections):
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
    

    def image_to_text(self, image: np.ndarray, boxes: Detections):
        """
        Extract the text from the detected bounding boxes
        using EasyOCR.

        Parameters:
            - `image`: OpenCV image
            - `boxes`: Detections

        Returns:
            - `List[str]`: List of extracted texts
        """
        texts = list()
        for box in boxes:
            text_box = image[box[1]:box[3], box[0]:box[2]]
            extract = self.reader.recognize(text_box, detail=0)
            texts.append(extract[0])
        return texts

    
    async def build_response(self, cv_image: Optional[np.ndarray] = None):
        # get the image in OpenCV format
        if cv_image is None:
            cv_image = await self.decode()

        # perform text deteciion
        detections = self.text_detection(cv_image)
        self.validate_inference(detections)

        # postprocess the detections
        boxes, labels = self.postprocess(detections)

        # extract the text from the image
        texts = self.image_to_text(cv_image, boxes)

        return {self.yolo.names[label]: text for label, text in zip(labels, texts)}
    