from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
from keras.models import Sequential
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import random
import cv2


# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths
Paths = os.listdir('./data_cleaned/')

# dictionary to map symbols to numbers for traing the model
label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
              '8': 8, '9': 9, '+': 10, '-': 11, 't': 12, 'l': 13, 'g': 14, 'n': 15}

# loop over the input images
for Path in Paths:
    print("images in: ", Path)

    for imagepath in os.listdir('./data_cleaned/'+Path):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread('./data_cleaned/'+Path+'/' +
                           imagepath, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = (255-image)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        data.append(image)

    # extract the class label from the image path and update the labels list
        label = imagepath[0]
        labels.append(label_dict[label[0]])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(x_train, x_test, y_train, y_test) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(16, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x_test, y_test)

# serialize model to JSON
model_json = model.to_json()
with open("../app/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
