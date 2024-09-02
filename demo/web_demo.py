# -*- encoding: utf-8 -*-
# @Time    :   2024/08/17 21:38:16
# @File    :   web_demo.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   web UI by gradio

import torch
import gradio as gr
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "outputs/weather_predict/checkpoint-113"
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
processor = AutoImageProcessor.from_pretrained(model_name)

@torch.inference_mode()
def inference(image_path, topk):
    topk = int(min(topk, len(model.config.id2label)))  # maybe topk great than the label num of model
    
    image = Image.open(image_path)
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs.to(device))
    logits = outputs.logits
    logits = F.softmax(logits, dim=-1)
    logits.squeeze_(0)  # remove batch
    score, index = torch.topk(logits, topk, dim=-1)
    score, index = score.tolist(), index.tolist()
    result = {}
    for i, s in zip(index, score):
        label = model.config.id2label[i]
        result[label] = s
    return result
    
    

iface = gr.Interface(
    fn=inference, 
    inputs=[gr.Image(type="filepath"), gr.Slider(0, 10, step=1, value=5, label="Top-K")],
    outputs=gr.Label(),
    title="image classification",
    description="upload a image to classify",
)

if __name__ == "__main__":
    iface.launch(server_port=2024)
