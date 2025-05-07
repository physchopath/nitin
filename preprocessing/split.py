import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models

# QHED function to enhance image quality
def apply_qhed(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    # Convert back to BGR format for the CNN
    return cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {label: idx for idx, label in enumerate(os.listdir(folder)) if os.path.isdir(os.path.join(folder, label))}
    
    for label in label_dict.keys():
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(img_path)
                if img is not None:
                    # Apply QHED preprocessing (contrast enhancement)
                    img = apply_qhed(img)
                    # Resize image to 48x48 if necessary
                    img = cv2.resize(img, (48, 48))
                    # Normalize the image
                    img = img / 255.0
                    images.append(img)
                    labels.append(label_dict[label])
    return np.array(images), np.array(labels)

# Replace this with your dataset directory
dataset_path = 'model/ck+/'
images, labels = load_images_from_folder(dataset_path)

print(f'Loaded {len(images)} images.')
print(f'Shape of images: {images.shape}')
print(f'Shape of labels: {labels.shape}')

# One-hot encode the labels
labels = to_categorical(labels)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f'Training set shape: {X_train.shape}, {y_train.shape}')
print(f'Testing set shape: {X_test.shape}, {y_test.shape}')

# Define the CNN model architecture
model = models.Sequential([
    layers.InputLayer(input_shape=(48, 48, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # Output layer for 7 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model after training
model.save('preprocessing/my_trained_model_qhed.h5')
print("Model saved as 'my_trained_model_qhed.h5'")
