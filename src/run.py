import gesture_dataset
from PIL import Image
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from tensorflow.keras.layers import (
    Layer,
    GlobalAveragePooling2D,
    Conv2D,
    MaxPool2D,
    Dense,
    Flatten,
    InputLayer,
    BatchNormalization,
    Input,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomContrast,
    RandomBrightness,
    Resizing,
    Rescaling
)
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam




# Load the dataset
dataset_path = "/home/shima/Dataset"
output_directory = "dataset"

gesture_dataset.load_dataset(dataset_path, output_directory)



def prepare_dataset(output_directory):
    """
    recognize train_images and train_labels
    """
    train_images = []
    train_labels = []


    for image_path in os.listdir(output_directory):
        image = Image.open(os.path.join(output_directory, image_path))
        image_array = np.array(image)
        train_images.append(image_array)
        train_labels.append(image_path.split("_")[0])
        
    return train_images, train_labels
        

def display(train_images, train_labels, num_images_to_display):
    """""
    Display 4 images before augmentation

    """
    # shuffle
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
    # Create a new figure
    fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

    for i in range(num_images_to_display):
    # Display each image in a subplot
        ax = plt.subplot(1, num_images_to_display, i + 1)
        ax.imshow(train_images[i])
        ax.set_title(f"Label: {train_labels[i]}")
        ax.axis('off')  # Turn off axis labels
    plt.show()
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig("images.png")
    plt.close()
    

# Prepare the dataset
train_images, train_labels = prepare_dataset(output_directory)
 
# Display the images
display(train_images, train_labels, num_images_to_display=4)



# Resize anb Normalization the images
train_images = np.array([np.array(Image.fromarray(image_array).resize((200, 200))).astype('float32') / 255.0 for image_array in train_images])

# Convert the labels to a NumPy array
train_labels = np.array(train_labels)

# split train images and validation images
train_images, test_images, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42  
)

# Further split the test set into test and validation sets
test_images, val_images, test_labels, val_labels = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=42
)


num_classes = len(set(train_labels))
print(f"Number of classes: {num_classes}")

# One hot encodeing for train_labels and val_labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=4)

val_labels_encoded = label_encoder.transform(val_labels)
val_labels_one_hot = to_categorical(val_labels_encoded, num_classes=4)


# Apply data augmentation 
train_datagen_augmented = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data_augmented = train_datagen_augmented.flow(
    train_images, train_labels_one_hot, batch_size=64
)
val_datagen_augmented = ImageDataGenerator()   
val_data_augmented = val_datagen_augmented.flow(
    val_images, val_labels_one_hot, batch_size=32
)


# Model
# transfer learning
weights_path = "/home/shima/efficientnetv2-b0_notop.h5"
backbone = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights=weights_path,
    input_shape=(200, 200, 3),
    include_preprocessing=True
)

backbone.trainable = False

efficient_model = tf.keras.Sequential([
    Input(shape=(200, 200, 3)),
    backbone,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

efficient_model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss = CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Model Training 
efficient_history = efficient_model.fit(
    train_data_augmented,
    validation_data = val_data_augmented,
    epochs = 10,
    verbose = 1
)
