import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet50BinaryClassifier, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Binary classification output

    def forward(self, x):
        return self.model(x)

model = ResNet50BinaryClassifier()
try:
    model.load_state_dict(torch.load('cervical_cancer_detection_binary_resnet50.pth', map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
