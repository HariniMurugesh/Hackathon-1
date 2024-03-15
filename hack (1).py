#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[2]:


# Function to load images and labels from a folder
def load_images_and_labels_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            # Load image
            img = cv2.imread(os.path.join(folder, filename))
            images.append(img)
            # Extract label from filename
            label = filename.split('_')[0]  # Assuming the label is the first part of the filename before '_'
            labels.append(label)
    return images, labels


# In[3]:


# Preprocessing function to resize images
def preprocess_images(images, target_height, target_width):
    preprocessed_images = []
    for img in images:
        # Resize image to target dimensions
        resized_img = cv2.resize(img, (target_width, target_height))
        # Other preprocessing steps can be added here
        # For example, converting to grayscale, normalization, etc.
        preprocessed_images.append(resized_img)
    return preprocessed_images


# In[4]:


# Load Images and Labels
folder_path = 'C:\\Users\\Harini\\OneDrive\\Hackathon cheques\\Images'
cheque_images, labels = load_images_and_labels_from_folder(folder_path)
print("Number of loaded cheque images:", len(cheque_images))
print("Number of loaded labels:", len(labels))
num_classes = len(set(labels))


# In[5]:


# Verify if any images were loaded
if len(cheque_images) == 0:
    print("No images found in the specified folder:", folder_path)
else:
    # Preprocess Images
    target_height = 224  # Specify the desired height for preprocessed images
    target_width = 224   # Specify the desired width for preprocessed images
    preprocessed_images = preprocess_images(cheque_images, target_height, target_width)
    print("Number of preprocessed images:", len(preprocessed_images))
    
    


# In[6]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)

# Fit and transform the label encoder (on both training and test labels)
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()


# In[7]:


# Combine training and test labels to fit the encoder on the entire label set
all_labels = y_train + y_test

# Fit the label encoder on all labels
label_encoder.fit(all_labels)


# In[8]:



# Transform both training and test labels
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Normalize pixel values to [0, 1]
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0


# In[9]:


# Define your neural network architecture
num_classes = len(set(labels))  # Assuming the number of classes is the number of unique labels
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(target_height, target_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# In[10]:


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[11]:


model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


# In[12]:


# Data augmentation
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)


# In[13]:


# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)


# In[14]:


# Train the model with data augmentation and early stopping
history = model.fit(train_datagen.flow(X_train, y_train_encoded, batch_size=32),
                    epochs=100,
                    validation_data=(X_test, y_test_encoded),
                    callbacks=[early_stopping])


# In[15]:


# Evaluate model performance on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[16]:


# (Your code for loading images, preprocessing, splitting data, defining the model, and training)

# Evaluate model performance on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Visualize training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[17]:


# Define target image dimensions
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

target_height = 224  # Specify the desired height for preprocessed images
target_width = 224   # Specify the desired width for preprocessed images


# In[18]:


# Load pre-trained VGG16 model
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(target_height, target_width, 3))

# Freeze convolutional base
vgg_base.trainable = False


# In[19]:


# Create a new model with VGG16 base
model_vgg = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the VGG16-based model
# Compile the VGG16-based model
model_vgg.compile(optimizer=RMSprop(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# In[20]:


# Define callbacks (e.g., checkpointing)
checkpoint = ModelCheckpoint('best_model_vgg.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# In[21]:


# Train the VGG16-based model
history_vgg = model_vgg.fit(train_datagen.flow(X_train, y_train_encoded, batch_size=16),
                             epochs=50,
                             validation_data=(X_test, y_test_encoded),
                             callbacks=[early_stopping, checkpoint])


# In[30]:


# Evaluate VGG16-based model
test_loss_vgg, test_accuracy_vgg = model_vgg.evaluate(X_test, y_test_encoded)
print("VGG16 Test Loss:", test_loss_vgg)
print("VGG16 Test Accuracy:", test_accuracy_vgg)


# In[31]:


# Plot training history for VGG16-based model
plt.plot(history_vgg.history['accuracy'], label='VGG16 Training Accuracy')
plt.plot(history_vgg.history['val_accuracy'], label='VGG16 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('VGG16 Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history_vgg.history['loss'], label='VGG16 Training Loss')
plt.plot(history_vgg.history['val_loss'], label='VGG16 Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VGG16 Training and Validation Loss')
plt.legend()
plt.show()


# In[32]:


# Save the model
model.save("your_model_name.h5")


# In[33]:


# Define a custom preprocessing function
def preprocess_function(image):
    # Perform preprocessing steps on the image
    # Example: Normalize pixel values to [0, 1]
    image = image / 255.0
    return image

# Create an ImageDataGenerator with the custom preprocessing function
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Example: Adjust brightness
    preprocessing_function=preprocess_function  # Use custom preprocessing function
)

# Example of custom preprocessing function
def preprocess_function(image):
    # Preprocess image here
    return image


# In[34]:


from sklearn.metrics import classification_report, confusion_matrix

# Evaluate model performance on the test dataset
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred_classes))


# In[28]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




