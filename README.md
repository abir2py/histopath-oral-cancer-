ORCANET: Oral Cancer Detection using Machine Learning
ORCANET is a web-based application built using Streamlit and PyTorch to classify histopathological images of the oral cavity into Leukoplakia or Oral Squamous Cell Carcinoma (OSCC). The app features a beautiful UI with video background, real-time predictions, and a visual pie chart representation of class probabilities.

🧠 Model
This application uses a ResNet-50 deep learning model with a custom fully connected classifier head:

Pretrained on: Custom dataset (not ImageNet)

Final layers:

Linear → ReLU → Dropout

Linear → ReLU

Linear → Sigmoid (for binary classification)

🔧 Features
Upload an image (PNG, JPG, JPEG)

Get real-time predictions: Leukoplakia or OSCC

Probability scores for both classes

Beautiful pie chart visualization

Elegant UI with video background and styled components

🖼️ Input
Accepts RGB images (converted automatically)

Resize and normalize images to match ResNet-50 input standards

📊 Output
Predicted Class: Leukoplakia or OSCC

Probability of each class

3D Pie Chart showing class probabilities
