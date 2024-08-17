# -*- encoding: utf-8 -*-
# @Time    :   2024/08/17 21:38:16
# @File    :   web_demo.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   web UI by gradio


import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch


model_name = "outputs/weather_predict/checkpoint-112"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

@torch.inference_mode()
def predict(image):
    inputs = processor(images=image, return_tensors="pt")
    
    outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label

iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="image classification",
    description="upload a image to classify",
)

if __name__ == "__main__":
    iface.launch(server_port=2024)
