from gradio import Interface
from gradio.components import JSON, Image

from src.api.handlers import InferenceHandler

handler = InferenceHandler(None)

iface = Interface(
    fn=handler.build_response,
    inputs=Image(type="numpy", label="Input Image"),
    outputs=JSON(label="Output")
)
