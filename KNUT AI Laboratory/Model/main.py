# -*- coding: utf-8 -*-
"""main

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b1RyYiAyAuLffUdk5RddPrV4JOQdyAgt

# Sector : Electricity Car-Battery

## Task : 2D Image-Classification

Training Accuracy Max : 98.77%

Training Accuracy Mean : 97.2%

Validation Accuracy Max : 91.13%

Validation Accuracy Mean : 83.7%
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.keras import datasets
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet101
from keras.models import Model

# Total Dataset(image Aug O)
# Approximately 20700 (Train 19050, Valid 1574)

num_classes = 5
image_size = 224

data_generator=ImageDataGenerator(preprocessing_function=preprocess_input,)

"""# Train_set"""

trainDir = '/content/drive/MyDrive/dataset4/train'
train_set = data_generator.flow_from_directory(
    trainDir,
    target_size=(image_size, image_size),
    batch_size=5000,
    class_mode='categorical',
    shuffle =True
    )

x_train, y_train = next(train_set)
len(x_train)

"""# Validation_set"""

testDir = '/content/drive/MyDrive/dataset4/test'
test_set = data_generator.flow_from_directory(
    testDir,
    target_size=(image_size, image_size),
    batch_size=620,
    class_mode='categorical',
    shuffle = False
    )

x_test, y_test = next(test_set) # next(): 반복 가능한 다음 객체 반환
len(x_test)

"""# Model (Pre-trained : ResNet101)
Tensorflow.keras
"""

base_model= ResNet101(weights='imagenet',pooling ="max" , include_top=False, input_shape=(224, 224, 3))

# Pre-trained Model Weight-freezing
for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('conv5_block3_out')
print(last_layer.output_shape)

# Downstream Task(2D Image-Classification)
x = Flatten()(last_layer.output) # Transform Vector

# Fully-connected
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)

# Multi-Classification Output Layer
x = Dense(5, activation = 'softmax')(x)

model = Model(base_model.input, x)

model.summary()

optimizer = Adam(lr=1e-6) # 자연상수(e) : 2.718
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist=model.fit(x_train, y_train, batch_size=120, epochs=20, validation_split=0.2)
#include_top=False, pooling=max ,v_split: 주어진 dataset을 설정한 hyper parameter값으로 train,test 비율로 나눠 평가

"""# Results"""

# Training Accuracy Mean : 97.2%
res = model.evaluate(x_train, y_train)
print(f'정확도={res[1]*100:.1f}%')

# Validation Accuracy Mean : 83.7%
res = model.evaluate(x_test, y_test)
print(f'정확도={res[1]*100:.1f}%')

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('ResNet101')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validataion'], loc = 'lower right')
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('ResNet101')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validataion'], loc = 'lower right')
plt.grid()
plt.show()

from tensorflow.python.keras.models import load_model
model.save('RN101_model.h5')

load_model = tf.keras.models.load_model('RN101_model.h5')