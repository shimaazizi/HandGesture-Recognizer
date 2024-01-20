import gesture_dataset
from PIL import Image
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

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
train_images = np.array([np.array(Image.fromarray(image_array).resize((200, 200))).astype('float32') / 255.0 for image_array in train_images])

#train_images = np.array([np.array(Image.fromarray(image_array).resize((200, 200)).convert('RGB')) for image_array in train_images])
#train_images = np.array([Image.fromarray(image_array).resize((200, 200)) for image_array in train_images])

# Convert the images to a NumPy array
#train_images = np.array(train_images)

# Normalize the images
#train_images = train_images / 255.0

# Add the color channel dimension
#train_images = train_images[:, :, :, np.newaxis]
# Convert the labels to a NumPy array
train_labels = np.array(train_labels)

# split train images and validation images
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42  
)

num_classes = len(set(train_labels))
print(f"Number of classes: {num_classes}")

# one hot encodeing for labels
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=4)

val_labels_encoded = label_encoder.transform(val_labels)
val_labels_one_hot = to_categorical(val_labels_encoded, num_classes=4)




#  create model 

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels_one_hot, epochs=5,
                    validation_data=(val_images, val_labels_one_hot))



# plot the accuracy and loss

def plot_and_save_curves(history, filename="curves.png"):
    """
    Plots and saves separate loss and accuracy curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss and accuracy on the same figure
    
    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.savefig(filename.replace(".png", "_loss.png"))
    plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    
    # Save the figure
    plt.savefig(filename.replace(".png", "_accuracy.png"))
    plt.close()

plot_and_save_curves(history, filename="curves.png")