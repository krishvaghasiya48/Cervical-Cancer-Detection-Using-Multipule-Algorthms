# Cervical Cancer Detection System

## Overview
This project implements a cervical cancer detection system using deep learning and machine learning techniques. It includes data preparation, feature extraction, model training, ensemble classification, and a graphical user interface (GUI) for image analysis.

## Project Structure
- `prepare_data.py`: Prepares the binary classification data directory structure and optionally copies sample images.
- `prepare_data_binary.py`: Copies images from original class folders into binary labeled folders (`cancer` and `no_cancer`).
- `run_feature_extraction.py`: Extracts features from training images and saves them for classifier training.
- `train_classifier.py`: Trains a logistic regression classifier on extracted features.
- `test_load_model.py`: Defines and loads a ResNet50-based binary classifier model.
- `nn.py`: Defines a simple CNN architecture for classification.
- `estan.py`: Defines the ESTAN model (ensemble of spatial and channel-wise attention networks).
- `ensemble.py`: Implements an ensemble classifier combining multiple classifiers with majority voting.
- `train_model.py`: Main training script supporting multiple models (ResNet50, VGG19, CYENET, ESTAN) with data augmentation, weighted sampling, focal loss, early stopping, and checkpointing.
- `gui_full.py`: Tkinter-based GUI application for uploading cervical cell images and displaying cancer detection results.
- `retrain_model.bat`: Batch script to retrain the model by running `train_model.py`.

## Algorithms Used
- Logistic Regression (in `train_classifier.py`)
- Simple Convolutional Neural Network (CNN) (in `nn.py`)
- ResNet50 (pretrained on ImageNet, fine-tuned for binary classification)
- VGG19 (pretrained on ImageNet, fine-tuned for binary classification)
- CYENET (custom CNN architecture defined in `train_model.py`)
- ESTAN (Ensemble of Spatial and Channel-wise Attention Networks, defined in `estan.py`)
- Ensemble Classifier combining Logistic Regression, SVM, Random Forest, Gradient Boosting, AdaBoost, K-Nearest Neighbors, and ANN (in `ensemble.py`)


## Setup and Usage

### Data Preparation
1. Organize your dataset with images categorized into class folders (e.g., `im_Dyskeratotic`, `im_Koilocytotic`, etc.).
2. Run `prepare_data_binary.py` to create binary labeled folders (`cancer` and `no_cancer`) and copy images accordingly.
3. Alternatively, use `prepare_data.py` to create the binary data structure and copy sample images for testing.

### Feature Extraction and Classifier Training
- Run `run_feature_extraction.py` to extract features from training images.
- Run `train_classifier.py` to train a logistic regression classifier on the extracted features.

### Model Training
- Use `train_model.py` to train deep learning models. Supported models include ResNet50, VGG19, CYENET, and ESTAN.
- Adjust parameters such as model selection, batch size, epochs, and learning rate in the script.
- The script automatically splits data into training and validation sets if not already split.
- Model checkpoints are saved as `cervical_cancer_detection_binary_<model_name>.pth`.

### GUI Application
- Run `gui_full.py` to launch the GUI.
- Upload cervical cell images for analysis.
- The GUI displays prediction probabilities, cancer risk assessment, and analysis conclusions.
- The model file `cervical_cancer_detection_binary_resnet50.pth` must be present in the project directory.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- numpy
- Pillow
- tkinter (usually included with Python)
- splitfolders
- tensorboard

Install dependencies using:
```
pip install torch torchvision scikit-learn numpy pillow splitfolders tensorboard
```

## Notes
- The system supports binary classification: Cancer vs No Cancer.
- Data augmentation and focal loss are used to improve model robustness.
- The ensemble classifier combines multiple traditional ML classifiers with an ANN.
- The GUI provides a user-friendly interface for image analysis and risk assessment.

## Running the Project
1. Prepare your data as described.
2. Train the model using `train_model.py` or use the pretrained model if available.
3. Launch the GUI with `gui_full.py` and upload images for detection.

## License
This project is provided as-is for research and educational purposes.
