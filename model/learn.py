import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import cv2

# Parameters
n_qubits = 6
n_classes = 7
epochs = 30
batch_size = 16

# 1. Load and preprocess CK+ dataset
def load_ck_dataset(data_dir):
    X, y = [], []
    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        for img_file in os.listdir(emotion_dir):
            img = cv2.imread(os.path.join(emotion_dir, img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48)) / 255.0
            X.append(img.flatten()[:n_qubits])  # Use only first 6 features
            y.append(emotion)
    return np.array(X), np.array(y)

X, y = load_ck_dataset("model/ck+/")  # <-- Your dataset path

# 2. Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=n_classes)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, stratify=y)

# 4. Define quantum circuit
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 5. Wrap as Keras layer
weight_shapes = {"weights": (6, n_qubits, 3)}
qlayer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# 6. Build hybrid model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(n_qubits,)),
    qlayer,
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(n_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 7. Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

# 8. Save model and encoder
model.save("model/qhed_model.h5")
np.save("model/label_classes.npy", encoder.classes_)
