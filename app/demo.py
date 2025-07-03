import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path().resolve() / "scripts"))
from train    import get_model, IMG_SIZE, DEVICE
from sign_names import SIGN_NAMES

@st.cache_resource
def load_model():
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load("models/traffic_resnet18.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

st.title("Traffic Sign Recognition Demo")

uploaded = st.file_uploader("Upload a traffic sign image", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize(IMG_SIZE)
    st.image(img, caption="Input Image", use_column_width=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    inp = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(inp).argmax(1).item()

    st.success(f"Predicted: **{SIGN_NAMES[pred]}** (class {pred})")