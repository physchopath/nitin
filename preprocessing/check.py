import os
import cv2
import numpy as np

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
                    images.append(img)
                    labels.append(label_dict[label])
    return np.array(images), np.array(labels)

# Replace this with your dataset directory
dataset_path = 'model/ck+/'
images, labels = load_images_from_folder(dataset_path)

print(f'Loaded {len(images)} images.')
print(f'Shape of images: {images.shape}')
print(f'Shape of labels: {labels.shape}')
