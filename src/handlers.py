import os

import cv2
import numpy as np
import torch
import easyocr
from fastapi import HTTPException, UploadFile, status
from supervision import Detections
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class InferenceHandler:

    __slots__ = ("image", "yolo", "reader")

    @staticmethod
    def download_assets():
        """
        Download the assets required for the inference
        """
        # download the YOLO model
        hf_hub_download(
            repo_id = "arnabdhar/YOLOv8-nano-aadhar-card",
            filename = "model.pt",
            local_dir = "./models"
        )

        # download the EasyOCR model
        _ = easyocr.Reader(["en"], gpu=False, detector=False, recognizer=True, verbose=False)

        return None
        

    def __init__(self, image: UploadFile):
        self.image = image
        self.yolo = YOLO("./models/model.pt")
        self.reader = easyocr.Reader(["en"], gpu=False, detector=False, recognizer=True, verbose=False)


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
            confidence = float(os.getenv("YOLO_CONFIDENCE", "0.6")),\
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

    
    async def build_response(self):
        # get the image in OpenCV format
        cv_image = await self.decode()

        # perform text deteciion
        detections = self.text_detection(cv_image)
        self.validate_inference(detections)

        # postprocess the detections
        boxes, labels = self.postprocess(detections)

        # extract the text from the image
        texts = self.image_to_text(cv_image, boxes)

        return {self.yolo.names[label]: text for label, text in zip(labels, texts)}
    