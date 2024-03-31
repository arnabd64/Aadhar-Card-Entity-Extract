import os
from typing import Annotated

import numpy as np
import psutil
from fastapi import Form, UploadFile

from src.core import decode, pipeline


class InferenceAPIHandler:

    __slots__ = ("image", "confidence")

    def __init__(self, image: UploadFile, confidence: Annotated[float, Form(...)] = 0.6):
        self.image = image
        self.confidence = float(confidence)


    async def response(self):
        """
        Get the response from the pipeline
        """
        cv_image = decode(await self.image.read())
        return pipeline(cv_image, self.confidence)
    

class APIHealthHandler:

    def __init__(self):
        pass


    def response(self):
        """
        Get the health metrics of the API
        """
        return {
            "status": "UP",
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage(os.getcwd()).percent
        }
    

class WebAppHandler:

    def response(self, image: np.ndarray, confidence: float = 0.6):
        """
        Get the response from the pipeline
        """
        return pipeline(image, confidence)