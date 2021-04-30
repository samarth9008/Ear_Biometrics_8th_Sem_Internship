#importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
import cv2

#importing libararies for creating model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

#Appling Dataset and validation path
dataset_path = 'E:\\sem 8\\Datasets\\Wild'
train_path = 'E:\\sem 8\\Datasets\\Wild\\train'
test_path = 'E:\\sem 8\\Datasets\\Wild\\test'

#Getting number of folders in the directory
print (os.listdir(dataset_path))

IMG_WIDTH = 150
IMG_HEIGHT = 150

training_data = []
train_categories = os.listdir(train_path)
for category in train_categories:
    img_path = os.path.join(train_path, category)
    class_num = train_categories.index(category)
    for img in os.listdir(img_path):
        img_array = cv2.imread(os.path.join(img_path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        training_data.append([new_array, class_num])


random.shuffle(training_data)

a = []
b = []

for features, labels in training_data:
    a.append(features)
    b.append(labels)

a = np.array(a).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
print(a.shape)
np.random.seed(1000)

X_train = a
Y_train = b

X_train = X_train / 255.0

print (X_train)
print (Y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(164, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train, to_categorical(Y_train), batch_size=200, epochs=50)

testing_data = []
test_categories = os.listdir(test_path)
for category in test_categories:
    img_path = os.path.join(test_path, category)
    class_num = train_categories.index(category)
    for img in os.listdir(img_path):
        img_array = cv2.imread(os.path.join(img_path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        testing_data.append([new_array, class_num])


p = []      #feature set
q = []      #label
for features, labels in testing_data:
    p.append(features)
    q.append(labels)

p = np.array(p).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
print(p.shape)

X_test = p
Y_test = q

X_test = X_test/255.0

# Evaluate the model
scores = model.evaluate(X_test, to_categorical(Y_test))

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])