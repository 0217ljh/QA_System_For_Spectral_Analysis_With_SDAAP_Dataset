import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("character")
launch_gradio_widget(module)
