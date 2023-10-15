from gradio import Interface
from gradio.components import JSON, Image
from pipelines import main


app = Interface(
    fn = main,
    inputs = Image(source='upload', type='numpy'),
    outputs = JSON(),
    allow_flagging = 'never'
)

if __name__ == "__main__":
    app.launch(server_name = '0.0.0.0', server_port=8001)