from gradio import Interface
from gradio.components import JSON, Image, Slider

from src.handlers import WebAppHandler


input_image = Image(
    type = "numpy",
    label = "Photo of Card",
    show_label = True,
    height = 480,
    width = 640
)

input_confidence = Slider(
    minimum = 0.01,
    maximum = 1,
    value = 0.25,
    step = 0.01,
    show_label = True,
    label = "Prediction Confidence"
)

output_json = JSON(
    label = "Entities",
    show_label = True
)


app = Interface(
    fn = WebAppHandler().response,
    inputs = [input_image, input_confidence],
    outputs = output_json,
    allow_flagging = "never",
    concurrency_limit = 5,
    title = "Aadhar Card OCR",
    analytics_enabled = True    
)