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

df = pd.read_json(
    DATA_DIRECTORY + '/labels.json').T.set_index('filename')
df['state'] = df.file_attributes.apply(lambda x: x['state']
                                       if 'state' in x else None)

file_order = None
for r, d, f in os.walk(DATA_DIRECTORY + '/images'):
    file_order = f
    break

labels = df.state.reindex(file_order).dropna()

data = tf.keras.preprocessing.image_dataset_from_directory(DATA_DIRECTORY,
                                                           labels='inferred',
                                                           label_mode='categorical')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(256, 256)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
