
from pathlib import Path
import gradio as gr
import os
import torch

from model import create_effnetb2_model, create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

MODEL_TYPE = "vit_b_16"
IMAGE_SIZE = 224

MODEL_DICT = {
    "effnet_b2": "Model/effnetb2_eu_moths.pth",
    "vit_b_16":  "Model/vitb16_eu_moths.pth"
}
MODEL_FUNCTION = {
    "effnet_b2": create_effnetb2_model, 
    "vit_b_16":  create_vit_model
}

# Define model path
MODEL_PATH = MODEL_DICT.get(MODEL_TYPE)

# Setup examples
example_path = Path("examples")
examples = list(example_path.glob("*.jpg"))

# Setup class names
class_names = []

with open("class_names.txt", 'r') as f:
    for line in f.readlines():
        class_names.append(line.strip('\n'))


### 2. Model and transforms preparation ###

# Create model
model, transforms = MODEL_FUNCTION.get(MODEL_TYPE)(
    num_classes=200, 
)

# Load saved weights
model.load_state_dict(
    torch.load(
        f=MODEL_PATH,
        map_location=torch.device("cpu"),  # load to CPU
    )
)

# Adjust transforms
transforms.crop_size = IMAGE_SIZE
transforms.resize_size = IMAGE_SIZE


def predict(img) -> Tuple[dict, float]:
    pred_labels_and_probs = {}
    
    start_time = timer()
    X = transforms(img).unsqueeze(0)

    model.eval()

    with torch.inference_mode():
        pred_logit = model(X)
        pred_softmax = torch.softmax(pred_logit, dim=1).tolist()[0]

        for idx, name in enumerate(class_names):
            pred_labels_and_probs[name] = round(pred_softmax[idx], 4)

    time = round(timer() - start_time, 2)
    
    return pred_labels_and_probs, time


def main():
    # Create title, description and article strings
    title = "Moth classifier v0.1"
    description = "An Vision Transformer model trained on the EU Moths data set (https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/). Model architecture introduced in: https://arxiv.org/abs/2010.11929"
    article = ""

    # Create the Gradio demo
    demo = gr.Interface(fn=predict, 
                        inputs=gr.Image(type="pil"),
                        outputs=[gr.Label(num_top_classes=200, label="Predictions"), 
                                gr.Number(label="Prediction time (s)")], 
                        examples=examples,
                        title=title,
                        description=description,
                        article=article)

    demo.launch(debug=False, 
                share=True) 


if __name__ == "__main__":
    main()

