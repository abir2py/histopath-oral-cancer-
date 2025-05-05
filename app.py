import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Function to load and preprocess the model
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()  # Sigmoid for binary classification
    )
    state_dict = torch.load('resnet_model.pth')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Function to preprocess the image and make a prediction
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        prob_class_1 = output.item()  # Probability for class 1 (OSCC)
        prob_class_0 = 1 - prob_class_1  # Probability for class 0 (leukoplakia)
        
        # Determine the predicted class
        predicted_class = 'OSCC' if prob_class_1 > 0.5 else 'Leukoplakia'
    
    return predicted_class, prob_class_0, prob_class_1

# Function to plot a 3D pie chart with percentages
def plot_pie_chart(prob_class_0, prob_class_1, predicted_class):
    labels = ['Leukoplakia (class 0)', 'OSCC (class 1)']
    sizes = [prob_class_0, prob_class_1]
    colors = ['#66b3ff', '#99ff99']
    explode = (0.1, 0)  # "explode" the first slice (Leukoplakia)

    # Plot the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)
    plt.title(f'Prediction: {predicted_class}', fontsize=18, weight='bold', color='green')
    
    # Display the pie chart in Streamlit
    st.pyplot(plt)

# Streamlit app code
def main():
    # Custom CSS for the app to make it beautiful and add video background
    st.markdown("""
        <style>
        /* Styling for the background video */
        video#background-video {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -1;
            opacity: 0.5; /* Dim the background slightly */
        }

        .title {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
        }

        .header {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        
        /* Custom styling for the uploaded image */
        .stFileUploader {
            display: block;
            margin: 0 auto;
        }

        .stWrite {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add video background (make sure the video file is in the correct location)
    st.markdown("""
        <video id="background-video" autoplay loop muted>
            <source src="C:\\cancer\\mixkit-abstract-red-fabric-flowing-282-full-hd.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    """, unsafe_allow_html=True)

    # Title and subtitle
    st.markdown('<h1 class="title">ORCANET</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="title">Oral Cancer Detection using Machine Learning</h1>', unsafe_allow_html=True)

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload an image for classification", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open the image and convert to RGB
        image = Image.open(uploaded_file).convert('RGB')

        # Display the uploaded image with a nice border
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Prediction
        predicted_class, prob_class_0, prob_class_1 = predict_image(image, model)

        # Display prediction results with formatted styling
        st.markdown(f"### Prediction Results")
        st.write(f"**Predicted class:** {predicted_class}")
        st.write(f"**Probability of Leukoplakia (class 0):** {prob_class_0 * 100:.2f}%")
        st.write(f"**Probability of OSCC (class 1):** {prob_class_1 * 100:.2f}%")

        # Plot 3D Pie Chart
        plot_pie_chart(prob_class_0, prob_class_1, predicted_class)

    else:
        st.write("Please upload an image to get the prediction.")

if __name__ == '__main__':
    main()
