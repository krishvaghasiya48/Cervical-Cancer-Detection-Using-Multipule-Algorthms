from feature_extractor import process_folder
import os

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Extract features for training data
process_folder('data/train', 'models/features_train.npz')

print("Feature extraction complete.")

def process_folder(input_folder, output_file):
    # your code here
