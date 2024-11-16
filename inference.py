import torch
from torchvision import models, transforms
import json
from PIL import Image
from io import BytesIO
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import os

# Define the same model architecture as in training
def net():
    num_classes = 133
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer for the new number of classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    # Apply weight initialization
    model.fc.apply(initialize_weights)

    return model

# Load weights into the defined model
def model_fn(model_dir):
    # Load the model
    model = net()  # Ensure `net` is your model architecture function
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.eval()

    # Attempt to load the class_to_idx mapping
    class_to_idx_path = os.path.join(model_dir, "class_to_idx.json")
    if os.path.exists(class_to_idx_path):
        with open(class_to_idx_path, "r") as f:
            class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            model.idx_to_class = idx_to_class
            print(f"LEGC Loaded idx_to_class mapping: {model.idx_to_class}")
    else:
        print(f"LEGC Error: {class_to_idx_path} not found.")
        model.idx_to_class = None

    return model
# Initialize weights as in training
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def input_fn(request_body, request_content_type):
    try:
        print(f"LEGC request_content_type: {request_content_type}")
        if request_content_type == "application/json":
            # Handle JSON input with base64 image encoding
            data = json.loads(request_body)
            img_data = base64.b64decode(data["image"])
            image = Image.open(BytesIO(img_data)).convert("RGB")
        
        elif request_content_type == "application/x-image" or request_content_type == "image/jpeg":
            print("LEGC Processing binary image data")
            # Convert the binary image to a PIL Image
            image = Image.open(BytesIO(request_body)).convert("RGB")
        
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        
        print("LEGC Image successfully loaded and processed")
        
        # Preprocessing: Resize, normalize, and add batch dimension
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        return input_tensor

    except Exception as e:
        print(f"LEGC Error in input_fn: {e}")
        raise

# Define the prediction function
def predict_fn(input_data, model):
    try:
        # Ensure model is on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_data = input_data.to(device)
        
        # Perform inference
        with torch.no_grad():
            output = model(input_data)
            _, predicted_label = torch.max(output, 1)
        
        # Check if `idx_to_class` is set
        if not hasattr(model, 'idx_to_class') or model.idx_to_class is None:
            raise ValueError("Model does not have `idx_to_class` mapping.")
        
        # Map label to class name
        class_name = model.idx_to_class[predicted_label.item()]
        return {"predicted_label": predicted_label.item(), "class_name": class_name}

    except Exception as e:
        print(f"LEGC Error in predict_fn: {e}")
        raise
    
# Define the output formatting function
def output_fn(prediction, content_type='application/json'):
    try:
        if content_type == 'application/json':
            response = json.dumps({"predicted_label": prediction})
            print("LEGC Output formatted successfully.")
            return response
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        print(f"LEGC Error in output_fn: {e}")
        raise