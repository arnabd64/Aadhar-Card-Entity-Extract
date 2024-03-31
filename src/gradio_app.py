from gradio import Interface
from gradio.components import JSON, Image

from src.handlers import WebAppHandler

handler = WebAppHandler()

iface = Interface(
    fn=handler.response,
    inputs=Image(type="numpy", label="Input Image"),
    outputs=JSON(label="Output")
)
