import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


  # Define the category names
category_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0


# Reshape your training and testing data to include the channel dimension
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))




# Display the first 25 images from the training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1) 
    plt.xticks([]) 
    plt.yticks([])
    plt.grid(False) 
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary) 
    plt.xlabel(category_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Additional Conv layer
    MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout layer
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create an instance of ImageDataGenerator with some transformations
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator on your data
datagen.fit(train_images)

# Example to print some information about the generated data
for x_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=32):
    print('Batch image shape:', x_batch.shape)
    print('Batch label shape:', y_batch.shape)
    break  # Remove this break to see more batches

# Train the model using the generator
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
          steps_per_epoch=len(train_images) // 32, epochs=10)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Load and preprocess an image

# Load and preprocess an image
image_path = 'pants.jpeg'
img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize the image to 28x28 pixels
img_array = np.array(img)  # Convert

# Normalize the image data
img_array = img_array / 255.0

# Reshape the array for the model (adding batch dimension)
img_array = img_array.reshape(1, 28, 28)

# Make a prediction
predictions = model.predict(img_array)

# Find the index of the category with the highest probability
predicted_category_index = np.argmax(predictions[0])


# Print the predicted category
print("Predicted category:", category_names[predicted_category_index])