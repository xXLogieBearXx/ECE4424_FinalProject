import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Load the diagnoses from the Excel file
diagnosis_df = pd.read_excel("messidor_diagnoses.xlsx", header=None)  # Assuming no header

# Directory where the images are stored
image_dir = "messidor_images/"

# Initialize empty lists to store images and labels
images = []
labels = []

# Find the maximum width and height among all images
max_width = 0
max_height = 0

# Iterate through each row in the diagnosis DataFrame
for idx, _ in diagnosis_df.iterrows():
    # Construct the image filename using the row index starting at 1
    image_filename = f"{idx+1}.tif"
    image_path = os.path.join(image_dir, image_filename)
    
    # Output the current image being resized
    print(f"Resizing image {image_filename}...")
    
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    
    # Check if the image is successfully loaded
    if image is None:
        print(f"Error: Unable to load image {image_filename}")
        continue
    
    # Update maximum width and height
    height, width = image.shape
    max_width = max(max_width, width)
    max_height = max(max_height, height)

# Find the target size to fit all images while maintaining aspect ratio
aspect_ratio = max_width / max_height
if aspect_ratio > 1:
    target_width = 2304  # Maximum width among all images
    target_height = int(target_width / aspect_ratio)
else:
    target_height = 1536  # Maximum height among all images
    target_width = int(target_height * aspect_ratio)

# Iterate through each row in the diagnosis DataFrame
for idx, _ in diagnosis_df.iterrows():
    # Construct the image filename using the row index starting at 1
    image_filename = f"{idx+1}.tif"
    image_path = os.path.join(image_dir, image_filename)
    
    # Output the current image being resized
    print(f"Resizing image {image_filename}...")
    
    # Read the image
    image = cv2.imread(image_path) 
    
    # Check if the image is successfully loaded
    if image is None:
        print(f"Error: Unable to load image {image_filename}")
        continue
    
    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Store the resized image
    images.append(resized_image)
    
    # Store the label
    label = diagnosis_df.loc[idx, 0]  # Assuming the labels are in the first column
    labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
print("list converted to numpy arrays")

# Normalize the images
images = images / 255.0  # Normalize pixel values to [0, 1]
print("images normalized")

# Convert labels to categorical format
labels = to_categorical(labels, num_classes=4)  # Assuming the maximum diagnosis value is 3
print("labels converted")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=69)
print("data split")

# Build the CNN model
model = Sequential([
    Input(shape=(target_height, target_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes for 0, 1, 2, 3
])
print("model built")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("model built")

# Train the model
print("Training started...")
history = model.fit(X_train, y_train, epochs=1000, batch_size=1, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Overall Test accuracy:', test_acc)
print('Overall Test loss:', test_loss)

# Calculate test accuracy for each class separately
class_acc = {}
for i in range(4):
    class_indices = np.where(np.argmax(y_test, axis=1) == i)[0]
    class_images = X_test[class_indices]
    class_labels = y_test[class_indices]
    _, class_accuracy = model.evaluate(class_images, class_labels, verbose=0)
    class_acc[i] = class_accuracy
    print(f'Test accuracy for class {i}:', class_accuracy)

# Calculate test accuracy for binning (0) and (1, 2, 3)
bin_indices = [0, 1, 1, 1]  # Map classes 0 to 0, and classes 1, 2, 3 to 1
bin_test_labels = np.array([bin_indices[np.argmax(label)] for label in y_test])
bin_test_accuracy = model.evaluate(X_test, to_categorical(bin_test_labels))[1]
print('Binned Test accuracy:', bin_test_accuracy)
