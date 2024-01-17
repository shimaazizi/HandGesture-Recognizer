import gesture_dataset
from PIL import Image
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = "/home/shima/Dataset"
output_directory = "dataset"

gesture_dataset.load_dataset(dataset_path, output_directory)


# recognize train_images and train_labels
train_images = []
train_labels = []

for image_path in os.listdir(output_directory):
    image = Image.open(os.path.join(output_directory, image_path))
    image_array = np.array(image)
    train_images.append(image_array)
    train_labels.append(image_path.split("_")[0])



# shuffle
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)  

# Display a few images with labels
num_images_to_display = 5

# Create a new figure
fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

for i in range(num_images_to_display):
    # Display each image in a subplot
    ax = plt.subplot(1, num_images_to_display, i + 1)
    ax.imshow(train_images[i])
    ax.set_title(f"Label: {train_labels[i]}")
    ax.axis('off')  # Turn off axis labels

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure to a file
plt.savefig("images.png")
plt.close()


# Resize the images
train_images = np.array([np.resize(image_array, (200, 200)) for image_array in train_images])

# Convert the images to a NumPy array
train_images = np.array(train_images)

# Normalize the images
train_images = train_images / 255.0

# split train images and validation images
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42  
)


