from ultralytics import YOLO
from supervision import Detections
from huggingface_hub import hf_hub_download
from easyocr import Reader
import numpy as np
import torch
import cv2



class TextDetectionPipeline:
    
    def __init__(self, confidence=0.6, device='cpu'):
        # download the modell
        hf_config = dict(
            repo_id = "arnabdhar/YOLOv8-nano-aadhar-card",
            filename = "model.pt"
        )
        self.model = YOLO(hf_hub_download(**hf_config))
        
        # set compute device
        self.device = torch.device(device)
        
        # prediction confidence
        assert confidence > 0.0 and confidence < 1.0, "confidence should be between 0 and 1"
        self.conf = confidence
                
        # label_id to label_names
        self.id2label = self.model.names
        
        
    def __call__(self, image: np.ndarray):
        # perform inference
        detections = self.model.predict(
            source = image,
            device = self.device,
            conf = self.conf,
            verbose = False
        )
        
        # convert to Detections object
        detections = Detections.from_ultralytics(detections[0])
        
        # validate inference
        assert len(detections.class_id) > 0, "No entities found on the image (Try reducing confidence)"
        assert len(detections.class_id) < 5, "Too many entities detected (Try increasing confidence)"
        assert 0 in detections.class_id, "No Aadhar Number Detected"
        
        # convert tthe data types
        boxes = detections.xyxy.astype(np.uint16)
        labels = detections.class_id.astype(np.uint8)
        
        return boxes, labels
    
    
    
class TextRecognitionPipeline:
    
    def __init__(self):
        self.reader = Reader(['en'], gpu=False, detector=False, recognizer=True, verbose=False)
        
        
    def __call__(self, image:np.ndarray, boxes:Detections) -> list[str]:
        texts = list()
        for box in boxes:
            text_box = image[box[1]:box[3], box[0]:box[2]]
            extract = self.reader.recognize(text_box, detail=0)
            texts.append(extract[0])
        return texts
        
        
detection = TextDetectionPipeline()
recognition = TextRecognitionPipeline()

def main(image: np.ndarray) -> dict[str, str]:
    boxes, labels = detection(image)
    texts = recognition(image, boxes)
    return {detection.id2label[label]: text for label, text in zip(labels, texts)}

