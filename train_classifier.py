import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

# Load features and labels
# Change the path if needed
features_path = 'models/features_train.npz'
data = np.load(features_path)
X, y = data['features'], data['labels']

# Encode string labels if necessary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save classifier and label encoder
joblib.dump(clf, 'models/logreg_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Define data directories
base_dir = 'D:/projects/cervical canser detector/data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Print directory structure
for dirpath, dirnames, filenames in os.walk(base_dir):
    print(f'Found directory: {dirpath}')
    for dirname in dirnames:
        print(f'\tSubdirectory: {dirname}')
    for filename in filenames:
        print(f'\tFile: {filename}')