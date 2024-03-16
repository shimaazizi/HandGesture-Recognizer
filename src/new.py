import os
from PIL import Image
from tensorflow.keras.utils import Sequence
import numpy as np

class CustomDataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size=32, target_size=(128, 128)):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_files = []
        self.labels = []

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, index):
        batch_images = []
        batch_labels = []

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        for gesture_class in os.listdir(self.dataset_path):
            gesture_class_path = os.path.join(self.dataset_path, gesture_class)
            if os.path.isdir(gesture_class_path):
                for image_file in os.listdir(gesture_class_path)[start_index:end_index]:
                    image_path = os.path.join(gesture_class_path, image_file)
                    image = Image.open(image_path)
                    image = image.resize(self.target_size)
                    image_array = np.array(image)
                    batch_images.append(image_array)
                    batch_labels.append(gesture_class)

        return np.array(batch_images), np.array(batch_labels)


