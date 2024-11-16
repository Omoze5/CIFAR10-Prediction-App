import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18



st.markdown("<h2 style='text-align: center; color: blue;'>CIFAR-10 Prediction App</h2>", unsafe_allow_html=True)
st.markdown("The CIFAR-10 Prediction App is a simple machine learning application designed to classify images into one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.Users can upload an image through the app, which preprocesses it and uses a trained ResNet18 model to predict the class of the image. The app showcases how deep learning models can accurately analyze and recognize visual patterns from datasets like CIFAR-10, demonstrating the practical use of image classification in various applications.")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="missing ScriptRunContext")

# Load pre-trained model
class MyConvBlock(nn.Module):
    def __init__(self):
        super(MyConvBlock, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(3, 32,3),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
            
        nn.Conv2d(32, 64,3),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
            
        nn.Conv2d(64,128,3),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Flatten(),
            
        nn.Linear(128*2*2,64),
        nn.Linear(64,10)
        
        )
        

    def forward(self, x):
        return self.model(x)
    


model = MyConvBlock() 
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()  # Set model to evaluation mode

# CIFAR-10 class labels
labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Ensure the image has 3 channels (RGB) - for grayscale, convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize image to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize based on CIFAR-10
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax().item()

    # Display the result
    st.write(f"Predicted Class: {labels[prediction]}")
