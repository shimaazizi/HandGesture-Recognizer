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
        self.classes = sorted(os.listdir(dataset_path))

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_image_files = self.image_files[start_index:end_index]
        batch_labels = self.labels[start_index:end_index]

        batch_images = []
        for image_file in batch_image_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            # Normalize the image data
            image_array = np.array(image) / 255.0
            batch_images.append(image_array)

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        self.image_files = []
        self.labels = []
        for gesture_class in self.classes:
            gesture_class_path = os.path.join(self.dataset_path, gesture_class)
            if os.path.isdir(gesture_class_path):
                for image_file in os.listdir(gesture_class_path):
                    self.image_files.append(os.path.join(gesture_class, image_file))
                    self.labels.append(gesture_class)


dataset_path = "/home/shima/Dataset"
custom_generator = CustomDataGenerator(dataset_path, batch_size=32, target_size=(128, 128))




