import tensorflow as tf
from requests import get
import pandas as pd
# from imread import imread_from_blob
import os
from os.path import isdir

DATA_DIRECTORY = 'trafficstate'


def format_image_directory(df, dir):
    """Creates subdirectories based on label data for TensorFlow image dataset
    preprocessing to autodetect"""
    u = df.state.unique()
    for i in u[u != None]:
        if not isdir(dir + '/' + i):
            os.mkdir(dir + '/' + i)

    for index, row in df.iterrows():
        if not row.state:
            os.remove(dir + '/images/' + index)
        else:
            os.rename(dir + '/images/' + index,
                      '/'.join([dir, row.state, index]))


# image = imread_from_blob(res.content)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIRECTORY,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    seed=6604,
    subset='training'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIRECTORY,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    seed=6604,
    subset='validation'
)

for image_batch, labels_batch in train_data:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=(256, 256)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)
