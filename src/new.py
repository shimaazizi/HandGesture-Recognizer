import os
from PIL import Image
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split

class CustomDataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size=32, target_size=(128, 128),
                 rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                 zoom_range=0.2, horizontal_flip=True, val_split=0.1, test_split=0.1):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.val_split = val_split
        self.test_split = test_split
        self.image_files = []
        self.labels = []
        self.classes = sorted(os.listdir(dataset_path))

            # Populate image_files with image file paths
        for gesture_class in self.classes:
            gesture_class_path = os.path.join(self.dataset_path, gesture_class)
            if os.path.isdir(gesture_class_path):
                for image_file in os.listdir(gesture_class_path):
                    self.image_files.append(os.path.join(gesture_class, image_file))
                    self.labels.append(gesture_class)
                    
        # Split the data into train, validation, and test sets
        self.train_files, self.val_test_files = train_test_split(self.image_files, test_size=val_split + test_split, random_state=42)
        self.val_files, self.test_files = train_test_split(self.val_test_files, test_size=test_split / (val_split + test_split), random_state=42)

    def __len__(self):
        return len(self.train_files) // self.batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_image_files = self.train_files[start_index:end_index]
        batch_labels = [self.labels[self.image_files.index(file)] for file in batch_image_files]

        batch_images = []
        for image_file in batch_image_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            # Convert the image to numpy array
            image_array = np.array(image)
            # Apply data augmentation
            image_array = self.apply_augmentation(image_array)
            # Normalize the image data
            image_array = image_array / 255.0
            batch_images.append(image_array)

        return np.array(batch_images), np.array(batch_labels)

    def apply_augmentation(self, image_array):
        # Apply data augmentation
        datagen = ImageDataGenerator(rotation_range=self.rotation_range,
                                     width_shift_range=self.width_shift_range,
                                     height_shift_range=self.height_shift_range,
                                     zoom_range=self.zoom_range,
                                     horizontal_flip=self.horizontal_flip)
        image_array = datagen.random_transform(image_array)
        return image_array

    def on_epoch_end(self):
        np.random.shuffle(self.train_files)

    def get_validation_data(self):
        val_images = []
        val_labels = []
        for image_file in self.val_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            image_array = np.array(image) / 255.0
            val_images.append(image_array)
            val_labels.append(self.labels[self.image_files.index(image_file)])
        return np.array(val_images), np.array(val_labels)

    def get_test_data(self):
        test_images = []
        test_labels = []
        for image_file in self.test_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            image_array = np.array(image) / 255.0
            test_images.append(image_array)
            test_labels.append(self.labels[self.image_files.index(image_file)])
        return np.array(test_images), np.array(test_labels)

# Example usage
dataset_path = "/home/shima/Dataset"
custom_generator = CustomDataGenerator(dataset_path, batch_size=32, target_size=(128, 128),
                                       rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       val_split=0.1, test_split=0.1)





