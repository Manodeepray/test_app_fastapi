# import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
# import onnx
# import onnx2torch

import logging


logger = logging.getLogger(__name__)


import model_arch

# Load ONNX model
# onnx_model = onnx.load("./models/resnet18_v1.onnx")

# # Convert ONNX to PyTorch
# pytorch_model = onnx2torch.convert(onnx_model)

# # Example inference
# input_tensor = torch.randn(1, 1, 224, 224)
# with torch.no_grad():
#     output = pytorch_model(input_tensor)

# predicted_class = output.argmax(1).item()
# print(f"Predicted class: {predicted_class}")












# Load ONNX model

# Define the image transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)  
    ])





# def load_onnx_model():
    
#     onnx_model = onnx.load("./models/resnet18_v1.onnx")

#     # Convert ONNX to PyTorch
#     model = onnx2torch.convert(onnx_model)    
#     model.eval()
#     return model
    


# Load and preprocess the image
def preprocess_image(image):
    image = image.resize((28, 28)) 

    image = transform(image)
        
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 28, 28)
    # image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    print("Processed image shape:", image.shape)
    # print(image)
    
    print("processed image :",image.shape)
    return image

# Predict
def predict(image,model):
    input_data = preprocess_image(image)
    # print(input_data)
    
    
    
    with torch.no_grad():
        outputs = model(input_data)
        print(outputs)
        _, predicted_class = torch.max(outputs.data, 1)
        print(predicted_class)
        
        
        predicted_class_int = predicted_class.item()
        print(predicted_class_int)
    return predicted_class_int


def load_pytorch_model():
    # Instantiate the model (make sure to define Resnet18 class before loading)
    
    pth_file_path = "./models/resnet18_v1.pth"
    model = model_arch.Resnet18(n_classes=10)

    # Load the saved weights
    model.load_state_dict(torch.load(pth_file_path))
    model.eval()
    
    return model




if __name__ == "__main__":
    # Example usage
    image_path = "./input_digits/digit.jpg"
    image = Image.open(image_path).convert("L")

    # Load the model
    model = load_pytorch_model()
    # model = load_onnx_model()   
    # Predict
    predicted_class = predict(image, model)
    print(f"Predicted class: {predicted_class}")



















# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torchvision.models as models

# # Define the image transformation (same as during training)
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale
# ])

# # Load PyTorch model
# def load_pytorch_model():
#     # Example: If it's a ResNet18 model
#     model = models.resnet18()
#     num_classes = 10  # Update to match the number of output classes
#     model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the output size
    
#     # Load the state dictionary
#     model.load_state_dict(torch.load("./models/resnet18_v1.pth", map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # Load and preprocess the image
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert("L")  # Convert to grayscale
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     print("Processed image shape:", image.shape)
#     return image

# # Predict
# def predict(image_path, model):
#     input_data = preprocess_image(image_path)
    
#     with torch.no_grad():
#         print(input_data)
#         outputs = model(input_data)
#         print(outputs)
#         _, predicted_class = torch.max(outputs.data, 1)
#     return predicted_class.item()

