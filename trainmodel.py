import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Define Data Paths
folders = {
    'glioma': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\glioma',
    'meningioma': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\meningioma',
    'pituitary': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\pituitary',
    'notumor': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\notumor',
}

# Step 2: Load and Preprocess Images
def load_images(folder, label):
    data = []
    for file in os.listdir(folder):
        try:
            img_path = os.path.join(folder, file)
            image = load_img(img_path, target_size=(128, 128))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append((image, label))
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue
    return data

# Step 3: Load data
print("Loading images...")
data = []
label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

for label_name, label_id in label_map.items():
    data += load_images(folders[label_name], label_id)

if not data:
    raise ValueError("No images found. Check your dataset paths.")

np.random.shuffle(data)

X = np.array([img for img, label in data])
y = np.array([label for img, label in data])

# Step 4: Feature Extraction
print("Extracting features with VGG16...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

# Predict in batches to prevent memory overflow
batch_size = 32
features = []

for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    batch_features = feature_model.predict(batch)
    features.append(batch_features)

features = np.concatenate(features, axis=0)
X_flat = features.reshape(features.shape[0], -1)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Step 6: Train Classifier
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate
print("Evaluating model...")
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Step 7.1: Visualize Confusion Matrix
class_names = list(label_map.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 8: Save Model and Labels
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Model and label map saved successfully.")
