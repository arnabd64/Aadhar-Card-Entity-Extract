from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from pipelines import main
import cv2
import numpy as np


server = FastAPI()

@server.get("/")
def root():
    return JSONResponse({"status": "Server is Working"})


@server.post("/api/inference")
async def inference(image: UploadFile):
    # decode the image
    buffer = await image.read()
    buffer = np.frombuffer(buffer, 'uint8')
    cv_image = cv2.imdecode(buffer, cv2.IMREAD_ANYCOLOR)    

    # run it through inference pipeline
    try:
        response = main(cv_image)
        return JSONResponse(response)
    
    except AssertionError as assert_error:
        return JSONResponse({"status": str(assert_error)}, 204)
    
    except Exception as e:
        return JSONResponse({"status": str(e)}, 201)