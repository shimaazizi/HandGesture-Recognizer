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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow import keras
from sklearn.model_selection import KFold

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
    unique_image_filenames = set()

    for image_path in os.listdir(output_directory):
        if image_path not in unique_image_filenames:
            image = Image.open(os.path.join(output_directory, image_path))
            image_array = np.array(image)

            # Extract the class name from the image file name
            gesture_class = image_path.split("_")[0]

            train_images.append(image_array)
            train_labels.append(gesture_class)
            unique_image_filenames.add(image_path)

    return np.array(train_images), np.array(train_labels)

        

def display(train_images, train_labels, num_images_to_display):
    """
    Display 4 images before augmentation
    """
    unique_labels = set(train_labels)
    print(f"Unique Labels: {unique_labels}")
    
    # Shuffle and zip images with labels
    shuffled_data = list(zip(train_images, train_labels))
    np.random.shuffle(shuffled_data)
    
    # Extract shuffled images and labels
    shuffled_images, shuffled_labels = zip(*shuffled_data)
    
    # Create a new figure
    fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

    displayed_labels = set()

    for i, (image, label) in enumerate(zip(shuffled_images, shuffled_labels)):
        if label not in displayed_labels:
            # Display each image in a subplot
            ax = plt.subplot(1, num_images_to_display, len(displayed_labels) + 1)
            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis('off')  # Turn off axis labels
            displayed_labels.add(label)

        if len(displayed_labels) == num_images_to_display:
            break

    plt.show()
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig("images.png")
    plt.close()

    

# Prepare the dataset
train_images, train_labels = prepare_dataset(output_directory)
print("train_images.shape:")
print(train_images.shape)

# Display the images
display(train_images, train_labels, num_images_to_display=4)

# Resize anb Normalization the images
train_images = np.array([np.array(Image.fromarray(image_array).resize((128, 128))).astype('float32') / 255.0 for image_array in train_images])
print(train_images.shape)

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

# one hot encodeing for labels
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=4)
print("train_labels_one_hot:")
print(train_labels_one_hot.shape)

val_labels_encoded = label_encoder.transform(val_labels)
val_labels_one_hot = to_categorical(val_labels_encoded, num_classes=4)
print("val_labels_one_hot:")
print(val_labels_one_hot.shape)


test_labels_encoded = label_encoder.transform(test_labels)
test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=4)

train_datagen_augmented = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data_augmented = train_datagen_augmented.flow(
    train_images, train_labels_one_hot, batch_size=64, shuffle=False
)
val_datagen_augmented = ImageDataGenerator()  
val_data_augmented = val_datagen_augmented.flow(
    val_images, val_labels_one_hot, batch_size=32, shuffle=False
)

# Model
def create_model():
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(4, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the K-fold cross-validation parameters
k_folds = 5
epochs = 20
batch_size = 8

# Perform K-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True)
fold_accuracy = []
fold_loss = []

for train_index, val_index in kf.split(train_images):
    train_images_fold, val_images_fold = train_images[train_index], train_images[val_index]
    train_labels_fold, val_labels_fold = train_labels_one_hot[train_index], train_labels_one_hot[val_index]
    
    model = create_model()
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )
    
    history = model.fit(
        train_images_fold,
        train_labels_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_images_fold, val_labels_fold),
        callbacks=[early_stopping]
    )
    
    fold_accuracy.append(history.history['val_accuracy'])
    fold_loss.append(history.history['val_loss'])

# Calculate average accuracy and loss across folds
avg_accuracy = np.mean(fold_accuracy, axis=0)
avg_loss = np.mean(fold_loss, axis=0)

print("Average Validation Accuracy Across Folds:", avg_accuracy)
print("Average Validation Loss Across Folds:", avg_loss)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot)

# Define class names
class_names = ['Fist', 'OpenPalm', 'PeaceSign', 'Thumbsup']

# Predict labels for test images
predicted_labels = model.predict(test_images)
predicted_classes = np.argmax(predicted_labels, axis=1)

# Convert one-hot encoded labels back to categorical labels
actual_labels = np.argmax(test_labels_one_hot, axis=1)

# Display images alongside predicted and actual labels
num_images_to_display = min(4, len(test_images))  # Display maximum 4 images or the number of images in the test set

plt.figure(figsize=(10, 10))
for i in range(num_images_to_display):
    plt.subplot(2, 2, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"Predicted: {class_names[predicted_classes[i]]}, Actual: {class_names[actual_labels[i]]}")
    plt.axis('off')
plt.show()
plt.tight_layout()

# Save the figure to a file
plt.savefig("prediction.png")
plt.close()

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