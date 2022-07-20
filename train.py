import tensorflow as tf
from requests import get
import pandas as pd
# from imread import imread_from_blob
import os
from os.path import isdir
from datetime import time, datetime

DATA_DIRECTORY = 'trafficstate'


def format_image_directory(df, dir):
    """Creates subdirectories based on label data for TensorFlow image dataset
    preprocessing to autodetect"""
    if not isdir(dir + '/q'):
        os.mkdir(dir + '/q')
    if not isdir(dir + '/f'):
        os.mkdir(dir + '/f')

    for index, row in df.iterrows():
        if not row.state in ('q', 'f'):
            os.remove(dir + '/' + index)
        else:
            os.rename(dir + '/' + index,
                      '/'.join([dir, row.state, index]))


# image = imread_from_blob(res.content)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIRECTORY,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    seed=6604,
    subset='training'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIRECTORY,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    seed=6604,
    subset='validation'
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(10, 3, input_shape=(
        256, 256), bias_regularizer='l1'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(7, 3, bias_regularizer='l1'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(5, 3, bias_regularizer='l1'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

i = input('Save model? (Y/n)')
if not i or i[0].lower() == 'y':
    model.save('model')
