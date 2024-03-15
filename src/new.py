import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import random

class CustomDataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size=32, target_size=(128, 128)):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_files = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for gesture_class in os.listdir(self.dataset_path):
            gesture_class_path = os.path.join(self.dataset_path, gesture_class)
            if os.path.isdir(gesture_class_path):
                for image_file in os.listdir(gesture_class_path):
                    self.image_files.append(os.path.join(gesture_class_path, image_file))
                    self.labels.append(gesture_class)

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, index):
        batch_images = []
        batch_labels = []

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        for image_file, label in zip(self.image_files[start_index:end_index], self.labels[start_index:end_index]):
            image = Image.open(image_file)
            image = image.resize(self.target_size)
            image_array = np.array(image)
            batch_images.append(image_array)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)

    def display(self, num_images_to_display):
        """
        Display 4 images before augmentation
        """
        unique_labels = set(self.labels)
        print(f"Unique Labels: {unique_labels}")

        # Shuffle and zip images with labels
        shuffled_data = list(zip(self.image_files, self.labels))
        np.random.shuffle(shuffled_data)

        # Extract shuffled images and labels
        shuffled_images, shuffled_labels = zip(*shuffled_data)

        # Create a new figure
        fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

        displayed_labels = set()

        for i, (image_file, label) in enumerate(zip(shuffled_images, shuffled_labels)):
            if label not in displayed_labels:
                # Display each image in a subplot
                image = Image.open(image_file)
                ax = plt.subplot(1, num_images_to_display, len(displayed_labels) + 1)
                ax.imshow(image)
                ax.set_title(f"Label: {label}", fontsize=14, fontweight='bold')
                ax.axis('off')
                displayed_labels.add(label)

            if len(displayed_labels) == num_images_to_display:
                break

        plt.show()
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig("images.png")
        plt.close()

# ###########################
dataset_path = "/home/shima/Dataset"
batch_size = 32
custom_generator = CustomDataGenerator(dataset_path, batch_size=batch_size)
custom_generator.display(4)
###################################


# Normalization
all_images = []
for i in range(len(custom_generator)):
    # Load data
    batch_images, _ = custom_generator[i]
    
    # Accumulate images
    all_images.extend(batch_images)

# Convert the list of images to a numpy array
all_images = np.array(all_images)

all_images_normalized = all_images.astype('float32') / 255.0

print(all_images_normalized.shape)

