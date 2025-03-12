# import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
# import onnx
# import onnx2torch
import logging
import model_arch

logger = logging.getLogger(__name__)




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
    """
    Load and preprocess an image for model input.

    Args:
        image (PIL.Image): The input image to be processed.

    Returns:
        torch.Tensor: A preprocessed image tensor with shape (1, 28, 28), 
                      normalized and ready for model input.

    Steps:
        1. Resize the image to 28x28 pixels.
        2. Apply transformations (e.g., normalization) using `transform`.
        3. Convert the image to a PyTorch tensor and add a batch dimension.
    """
    image = image.resize((28, 28))  # Resize image to 28x28 pixels
    image = transform(image)  # Apply transformations (e.g., normalization)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    print("Processed image shape:", image.shape)  # Debugging: Print final tensor shape
    return image



# Predict
def predict(image, model):
    """
    Predict the class of an input image using a loaded trained model.

    Args:
        image (PIL.Image): The input image to be classified.
        model (torch.nn.Module): The laoded trained model used for prediction.

    Returns:
        int: The predicted class label as an integer.

    Steps:
        1. Preprocess the image using `preprocess_image`.
        2. Perform inference using the model (no gradient computation).
        3. Extract the predicted class from the model's output.
    """
    input_data = preprocess_image(image)  # Preprocess the image for model input

    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(input_data)  # Get model predictions
        _, predicted_class = torch.max(outputs.data, 1)  # Get the class with the highest score
        predicted_class_int = predicted_class.item()  # Convert tensor to integer

    return predicted_class_int





def load_pytorch_model():
    """
    Load a pre-trained PyTorch model from a saved state dictionary.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.

    Steps:
        1. Instantiate the model architecture (Resnet18 with 10 output classes).
        2. Load the model weights from the specified `.pth` file.
        3. Set the model to evaluation mode.
    """
    pth_file_path = "./models/resnet18_v2.pth"  # Path to the saved model weights
    model = model_arch.Resnet18(n_classes=10)  # Instantiate the model

    model.load_state_dict(torch.load(pth_file_path))  # Load the saved weights
    model.eval()  # Set the model to evaluation mode

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

