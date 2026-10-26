"""Gradio demo - Plant Disease Detection"""
import json

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms

with open("classes.json") as f:
    CLASSES = json.load(f)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    x = tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    return {CLASSES[i].replace("___", " - ").replace("_", " "): float(p)
            for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a leaf image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Plant Disease Detection",
    description="Upload a photo of a plant leaf to detect the disease (EfficientNetB0, PlantVillage).",
)

if __name__ == "__main__":
    demo.launch()
