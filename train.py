# https://github.com/corey-snyder/STREETS

#from imread import imread_from_blob
import tensorflow as tf
from requests import get
import pandas as pd

DATA_DIRECTORY = 'trafficstate'
#image = imread_from_blob(res.content)

#labels = pd.from_json()

data = tf.keras.preprocessing.image_dataset_from_directory(DATA_DIRECTORY,
                                                           labels=None)
