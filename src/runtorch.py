import os
import math
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
from os.path import join

class CustomDataGenerator(Dataset):
    def __init__(self, x, y, label_encoder, batch_size=32, target_size=(128, 128),
                 rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                 zoom_range=0.2, horizontal_flip=True, train=True):
        self.x = x
        self.y = y
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.train = train
        self.classes = set(item.split("_")[0] for item in os.listdir(os.path.dirname(x[0])))
        
        # Define data augmentation transforms
        self.transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip() if self.horizontal_flip else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(self.rotation_range) if self.rotation_range else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.x))
        
        batch_image_files_path = self.x[start_index:end_index]
        batch_labels = self.y[start_index:end_index]

        batch_images = []
        for image_path in batch_image_files_path:
            if not os.path.exists(image_path):
                print(f"Image path does not exist: {image_path}")
                continue

            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transforms(image)
                if image is not None:  # Ensure image loading and transformation succeeded
                    batch_images.append(image)
                else:
                    print(f"Failed to load or transform image: {image_path}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

        if not batch_images:
            print(f"Empty batch at index {index}, images paths: {batch_image_files_path}")
            raise ValueError(f"No images loaded for the batch at index {index}")

        batch_labels_encoded = self.label_encoder.transform(batch_labels)
        batch_labels_tensor = torch.tensor(batch_labels_encoded, dtype=torch.long)
        return torch.stack(batch_images), batch_labels_tensor

    
        
        
        
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=124, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return F.softmax(x, dim=1)


def create_model():
    model = CustomModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion



def dataset_split(dataset_path: str, test_split: float = 0.2, val_split: float = 0.2):
    images_files = []
    labels = []
    
    for item in os.listdir(dataset_path):
        item_path = join(dataset_path, item)
        if os.path.isdir(item_path):
            class_label = item  
            for image_file in os.listdir(item_path):
                image_path = join(item_path, image_file)
                images_files.append(image_path)
                labels.append(class_label)
    
    # Print total number of images and unique class labels
    print(f"Total number of images: {len(images_files)}")
    print(f"Unique class labels: {set(labels)}")
    
    # Split the data into train, validation, and test sets
    x_train, x_test, y_train, y_test = train_test_split(images_files, labels, test_size=test_split, random_state=42, stratify=labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42, stratify=y_train)

    # Print the number of samples in each split
    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    return x_train, y_train, x_val, y_val, x_test, y_test

#dataset path
dataset_path = "/home/shima/Dataset"

# Split the dataset
x_train, y_train, x_val, y_val, x_test, y_test = dataset_split(dataset_path)


if __name__ == '__main__':
    # Create CustomDataGenerator for training data
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_split(dataset_path)
    print("train:", Counter(y_train))
    print("val:", Counter(y_val))
    print("test:", Counter(y_test))
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_val + y_test)

    train_generator = CustomDataGenerator(x_train, y_train,
                                          label_encoder,
                                          train=True,
                                          batch_size=16,
                                          target_size=(128, 128),
                                          rotation_range=10,
                                          # width_shift_range=0.2,
                                          # height_shift_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=False)

    # Create CustomDataGenerator for validation data
    val_generator = CustomDataGenerator(x_val, y_val, label_encoder,
                                        train=False, batch_size=32, target_size=(128, 128))
    test_generator = CustomDataGenerator(x_test, y_test, label_encoder,
                                         train=False, batch_size=32, target_size=(128, 128))


     # Create the model, optimizer, and loss function
    model, optimizer, criterion = create_model()

    # Training loop
    num_epochs = 10  # Define number of epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_generator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_generator)
        train_accuracy = train_correct / len(x_train)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_generator:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_generator)
        val_accuracy = val_correct / len(x_val)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Testing
    model.eval()
    test_correct = 0

    with torch.no_grad():
        for inputs, labels in test_generator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / len(x_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    
  
