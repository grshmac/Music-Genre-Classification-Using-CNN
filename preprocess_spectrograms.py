import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define the path to the dataset
GTZAN_PATH = "GTZAN"
IMAGES_PATH = os.path.join(GTZAN_PATH, "images_original")
OUTPUT_SIZE = (128, 128)  # Resize all images to 128x128

# Initialize lists to hold data and labels
data = []   # List to store the spectrogram image arrays
labels = [] # List to store the genre labels corresponding to each image

# Load images and labels from the dataset
for genre_folder in os.listdir(IMAGES_PATH):  # Loop through each genre folder
    genre_path = os.path.join(IMAGES_PATH, genre_folder)
    if os.path.isdir(genre_path):  # Ensure it's a directory
        for image_file in os.listdir(genre_path):  # Loop through each image in the genre folder
            image_path = os.path.join(genre_path, image_file)
            if image_file.endswith('.png'):  # Check if the file is a PNG image
                # Open the image, resize it, and convert it to grayscale
                img = Image.open(image_path).convert('L')  # 'L' mode converts image to grayscale
                img = img.resize(OUTPUT_SIZE)  # Resize the image to 128x128

                # Convert the image to a numpy array and normalize pixel values to the range [0, 1]
                img_array = np.array(img) / 255.0
                
                # Append the image array to 'data' and the genre label to 'labels'
                data.append(img_array)       # Add the processed image array to 'data'
                labels.append(genre_folder)  # Add the genre name as the label

# Convert 'data' and 'labels' lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Reshape 'data' to add a channel dimension (1 for grayscale)
data = data.reshape((-1, OUTPUT_SIZE[0], OUTPUT_SIZE[1], 1))

# Split the data into training, validation, and test sets (70%/15%/15% split)
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, stratify=labels)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, stratify=temp_labels)

# Save the data splits for easy loading during training
np.savez("train_data.npz", data=train_data, labels=train_labels)
np.savez("val_data.npz", data=val_data, labels=val_labels)
np.savez("test_data.npz", data=test_data, labels=test_labels)

print("Data preprocessing complete and split into training, validation, and test sets.")
