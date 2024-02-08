from PIL import Image
import os

def load_dataset(dataset_path, output_directory):
    for gesture_class in os.listdir(dataset_path):
        gesture_class_path = os.path.join(dataset_path, gesture_class)
        if os.path.isdir(gesture_class_path):
            for image_file in os.listdir(gesture_class_path):
                image_path = os.path.join(gesture_class_path, image_file)

                # Read image using Pillow
                image = Image.open(image_path)

                # Process or display the image as needed
                resized_image = image.resize((200, 200))

                # Create the output directory if it doesn't exist
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # Save the processed image
    
                output_path = os.path.join(output_directory, image_file[:-4])
                

                resized_image.save(output_path, format="JPEG")




