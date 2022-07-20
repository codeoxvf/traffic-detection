import tensorflow as tf
from requests import get
import pandas as pd
import os
from os.path import isdir
from datetime import time, datetime

DATA_DIRECTORY = 'data'
MODEL_DIRECTORY = 'model'

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

model = tf.keras.models.load_model(MODEL_DIRECTORY)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

i = input('Save model? (Y/n)')
if not i or i[0].lower() == 'y':
    model.save(MODEL_DIRECTORY)
