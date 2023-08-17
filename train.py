"""
train.py <model name> [<epochs>]

Creates new model or trains existing one. Takes model name
(models/<model name>) as argument
"""

from sys import argv
if len(argv) == 1:
    print('No model name given')
    exit()

import tensorflow as tf
from os.path import isfile

DATA_DIRECTORY = 'data'
MODEL_DIRECTORY = 'models'
EPOCHS = 6 if len(argv) < 3 else argv[2]

model_path = MODEL_DIRECTORY + '/' + argv[1]
if not isfile(model_path + '/saved_model.pb'):
    # new model
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(10, 3, input_shape=(256, 256)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(10, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(10, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
else:
    model = tf.keras.models.load_model(model_path)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIRECTORY,
    labels='inferred',
    label_mode='categorical',
    class_names=['e', 'l', 'h'],
    validation_split=0.2,
    seed=6604,
    subset='training'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIRECTORY,
    labels='inferred',
    label_mode='categorical',
    class_names=['e', 'l', 'h'],
    validation_split=0.2,
    seed=6604,
    subset='validation'
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

while True:
    save = input('Save model? Y/N ').lower()
    if save == 'y':
        new_path = input('Model name (' + argv[1] + ')? ')
        model_path = (MODEL_DIRECTORY + '/' + new_path) if new_path else model_path
        model.save(model_path)
        break
    elif save == 'n':
        break