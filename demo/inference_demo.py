# -*- encoding: utf-8 -*-
# @Time    :   2024/08/17 21:26:07
# @File    :   inference_demo.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   inference_demo

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

img_path = "data/0.jpg"
image = Image.open(img_path)
weight_path = "outputs/weather_predict/checkpoint-113"
processor = AutoImageProcessor.from_pretrained(weight_path)  #! need preprocessor_config.json
model = AutoModelForImageClassification.from_pretrained(weight_path)

model.eval()
with torch.inference_mode():
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])