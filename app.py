import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from torchvision import transforms

@st.cache_resource
def load_trained_model():
    # Load PyTorch model on CPU
    return torch.load("skin_cancer_model.pth", map_location=torch.device("cpu"))

model = load_trained_model()
model.eval()

class_labels = [
    'Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
    'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma'
]

# Define PyTorch transforms
transform = transforms.Compose([
    transforms.Resize((75, 100)),  # Note: H x W in PyTorch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3)
])

st.title("Skin Cancer Detection")
st.write("Upload a skin lesion image and get the predicted class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
            confidence = torch.softmax(prediction, dim=1).max().item() * 100

    st.success(f"**Prediction:** {class_labels[predicted_class]} ({confidence:.2f}% confidence)")
