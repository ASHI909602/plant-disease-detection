
import json

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CONF_THRESHOLD = 0.60  # isse kam confidence par warning dikhegi

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")


@st.cache_resource
def load_model():
    with open("classes.json") as f:
        classes = json.load(f)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, classes


tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("🌿 Plant Disease Detection")
st.write("Upload a photo of a plant leaf to detect the disease. "
         "Model: EfficientNetB0 fine-tuned on PlantVillage (38 classes).")
st.info("Best results: a single leaf, plain background, good light, photo taken close up. "
        "Supported crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, "
        "Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.")

model, classes = load_model()
file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    with torch.no_grad():
        probs = torch.softmax(model(tf(image).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, 3)

    st.subheader("Prediction")
    if float(top.values[0]) < CONF_THRESHOLD:
        st.warning("The model is not confident about this image. It may be a crop or disease "
                   "it was not trained on, or the photo may be unclear. Treat the result below "
                   "as unreliable and try a clearer photo of a single leaf.")
    for p, i in zip(top.values, top.indices):
        name = classes[i].replace("___", " - ").replace("_", " ")
        st.write(f"**{name}**: {float(p) * 100:.1f}%")
        st.progress(float(p))

st.caption("Trained on lab images (PlantVillage); accuracy on real field photos will be lower. "
           "Not a substitute for expert diagnosis.")
