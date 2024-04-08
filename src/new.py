import os
from PIL import Image
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CustomDataGeneratorTrain(Sequence):
    def __init__(self, dataset_path, batch_size=32, target_size=(128, 128),
                 rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                 zoom_range=0.2, horizontal_flip=True, test_split=0.1):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.test_split = test_split
        self.image_files = []
        self.labels = []
        self.classes = sorted(os.listdir(dataset_path))

        # Populate image_files with image file paths and labels
        for gesture_class in self.classes:
            gesture_class_path = os.path.join(self.dataset_path, gesture_class)
            if os.path.isdir(gesture_class_path):
                for image_file in os.listdir(gesture_class_path):
                    self.image_files.append(os.path.join(gesture_class, image_file))
                    self.labels.append(gesture_class)

        # Split the data into train and test sets
        self.train_files, self.test_files = train_test_split(self.image_files, test_size=test_split, random_state=42)

        # One-hot encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

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

        # One-hot encode labels
        batch_labels_encoded = self.label_encoder.transform(batch_labels)
        batch_labels_one_hot = to_categorical(batch_labels_encoded, num_classes=len(self.classes))

        return np.array(batch_images), np.array(batch_labels_one_hot)

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

    def get_train_data(self):
        train_images = []
        train_labels = []
        for image_file in self.train_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            image_array = np.array(image) / 255.0
            train_images.append(image_array)
            train_labels.append(self.labels[self.image_files.index(image_file)])
        
        # One-hot encode train labels
        train_labels_encoded = self.label_encoder.transform(train_labels)
        train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=len(self.classes))

        return np.array(train_images), np.array(train_labels_one_hot)
    
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
        
        # One-hot encode test labels
        test_labels_encoded = self.label_encoder.transform(test_labels)
        test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=len(self.classes))

        return np.array(test_images), np.array(test_labels_one_hot)


class CustomDataGeneratorVal(Sequence):
    def __init__(self, dataset_path, batch_size=32, target_size=(128, 128)):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_files = []
        self.labels = []
        self.classes = sorted(os.listdir(dataset_path))

        # Populate image_files with image file paths and labels
        for gesture_class in self.classes:
            gesture_class_path = os.path.join(self.dataset_path, gesture_class)
            if os.path.isdir(gesture_class_path):
                for image_file in os.listdir(gesture_class_path):
                    self.image_files.append(os.path.join(gesture_class, image_file))
                    self.labels.append(gesture_class)

        # One-hot encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_image_files = self.image_files[start_index:end_index]
        batch_labels = [self.labels[self.image_files.index(file)] for file in batch_image_files]

        batch_images = []
        for image_file in batch_image_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            # Convert the image to numpy array
            image_array = np.array(image)
            # Normalize the image data
            image_array = image_array / 255.0
            batch_images.append(image_array)

        # One-hot encode labels
        batch_labels_encoded = self.label_encoder.transform(batch_labels)
        batch_labels_one_hot = to_categorical(batch_labels_encoded, num_classes=len(self.classes))

        return np.array(batch_images), np.array(batch_labels_one_hot)

    def get_validation_data(self):
        val_images = []
        val_labels = []
        for image_file in self.image_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            image_array = np.array(image) / 255.0
            val_images.append(image_array)
            val_labels.append(self.labels[self.image_files.index(image_file)])
        
        # One-hot encode validation labels
        val_labels_encoded = self.label_encoder.transform(val_labels)
        val_labels_one_hot = to_categorical(val_labels_encoded, num_classes=len(self.classes))

        return np.array(val_images), np.array(val_labels_one_hot)


def create_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3), strides=2),
        layers.Dropout(0.5),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', strides=2),
        layers.Dropout(0.5),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', strides=2),
        layers.Dropout(0.5),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        #layers.Dropout(0.5),
        #layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(4, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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

# Create CustomDataGenerator for training data
dataset_path = "/home/shima/Dataset"
train_generator = CustomDataGeneratorTrain(dataset_path, batch_size=32, target_size=(128, 128),
                                           rotation_range=20, width_shift_range=0.2,
                                           height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                           test_split=0.1)

# Create CustomDataGenerator for validation data
val_generator = CustomDataGeneratorVal(dataset_path, batch_size=32, target_size=(128, 128))

# Create the model
model = create_model()

# Train the model
history = model.fit(train_generator,
                    epochs=10,  
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator))

# Evaluate the model on the test set
test_images, test_labels_one_hot = train_generator.get_test_data()  
test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot)

# Evaluate the model on the training set
train_images, train_labels_one_hot = train_generator.get_train_data()
train_loss, train_accuracy = model.evaluate(train_images, train_labels_one_hot)

# Plot and save the curves
plot_and_save_curves(history, filename="curves.png")






