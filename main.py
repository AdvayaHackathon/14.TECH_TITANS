from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ----------------- Flask App -----------------
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- Data Loading & Preprocessing -----------------
metadata_path = r"C:\skin disease\archive\data\metadata.csv"
image_dir = r"C:\skin disease\archive\data\images"
img_size = 128

# Load metadata
df = pd.read_csv(metadata_path)

label_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
for col in label_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        print(f"Column '{col}' not found in DataFrame!")


def get_label(row):
    for col in label_cols:
        if row[col] == 1:
            return col
    return 'Unknown'


df['label'] = df.apply(get_label, axis=1)


def load_images(df, image_dir):
    images = []
    labels = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image'] + '.jpg')
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(row['label'])
    return np.array(images), labels


X, y = load_images(df, image_dir)
X = X / 255.0

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# ----------------- Model Definition -----------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ----------------- Training -----------------
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=3, batch_size=32)
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Save Model
model.save("skin_disease_model.keras")

# ----------------- Flask Route for Upload + Prediction -----------------


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Preprocess uploaded image
        img = cv2.imread(path)
        if img is None:
            return "Failed to read image"
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        model = load_model("skin_disease_model.keras")
        prediction = model.predict(img)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

        return f"Prediction: {predicted_label}"


# ----------------- Run Flask -----------------
if __name__ == '__main__':
    app.run(debug=True)
