import os
import cv2
import numpy as np
import tensorflow as tf

# Define the path to the CK+ dataset
dataset_path = "model/ck+/"  # Replace this with your actual dataset path

# Define target size for resizing
target_size = (224, 224)  # Resize to 224x224 for common models

# Function to load and preprocess images from the CK+ dataset
def load_and_preprocess_images(dataset_path, target_size=(224, 224)):
    images = []
    labels = []

    # Loop through the dataset and process images
    for label_dir in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_dir)

        # Check if it's a directory
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)

                # Load the image
                image = cv2.imread(img_path)
                if image is None:
                    continue  # Skip if image cannot be loaded

                # Resize the image
                image_resized = cv2.resize(image, target_size)

                # Normalize to range [0, 1] and convert to float32
                image_normalized = image_resized.astype('float32') / 255.0

                # Append the preprocessed image and label
                images.append(image_normalized)
                labels.append(label_dir)  # Using the directory name as the label

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Step 1: Load and preprocess all images
images, labels = load_and_preprocess_images(dataset_path, target_size)

# Step 2: Convert labels to categorical (if required)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Encode the labels (if using one-hot encoding)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Step 3: Check the shape and data type of the preprocessed data
print("Shape of the images:", images.shape)  # Should be (num_images, height, width, channels)
print("Shape of the labels:", labels_categorical.shape)  # Should be (num_images, num_classes)
print("Data type of the images:", images.dtype)  # Should be float32
print("Data type of the labels:", labels_categorical.dtype)  # Should be int32 or float32

# Step 4: Split into training and validation sets (optional)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Step 5: Batch the images using TensorFlow's Dataset API (for large datasets)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(32)

# Now you're ready to use these datasets for model training
