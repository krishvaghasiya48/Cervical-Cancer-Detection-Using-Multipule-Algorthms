import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# Load pre-trained ResNet152
model = models.resnet152(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final FC
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_t)
    return features.squeeze().numpy()

def process_folder(folder, out_file):
    features, labels = [], []
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        if not os.path.isdir(class_folder):
            continue
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                feat = extract_features(img_path)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    np.savez(out_file, features=np.array(features), labels=np.array(labels))

# Example usage (uncomment to run):
# process_folder('data/train', 'models/features_train.npz') 