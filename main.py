from sys import argv
from os.path import isfile
import numpy as np

MODEL_DIRECTORY = 'models'

if len(argv) == 1:
    print('No model name given')
    exit()

if not isfile('/'.join([MODEL_DIRECTORY, argv[1], 'saved_model.pb'])):
    print('Model does not exist')
    exit()

import tensorflow as tf
from numpy import expand_dims
from matplotlib import pyplot as plt
from imread import imread_from_blob
from requests import get
model_path = MODEL_DIRECTORY + '/' + argv[1]

# load images
res = get('https://api.data.gov.sg/v1/transport/traffic-images')
data = res.json()

if not data['api_info']['status'] == 'healthy':
    print('Traffic API unavailable')
    exit()

batch = []
for cam in data['items'][0]['cameras']:
    img = get(cam['image'])
    batch.append(imread_from_blob(img.content))

# load saved model
model_path = MODEL_DIRECTORY + '/' + argv[1]
saved_model = tf.keras.models.load_model(model_path)

model = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 256),
    saved_model
])

# run model
classes = ['empty', 'light', 'heavy']
result = np.array([model.predict_on_batch(expand_dims(i, 0))[0] for i in batch])
preds = np.argmax(result, axis=1)

# display images
for i in range(len(batch)):
    if i % 4 == 0:
        if not i == 0:
            plt.show()
            plt.close()

        fig = plt.figure()

    sub = fig.add_subplot(2, 2, (i % 4)+1)
    # + ': ' + image_names[i].split('/')[-1])
    sub.set_title('Image ' + str(i+1))
    sub.set_xlabel('Prediction: ' + classes[preds[i]])

    sub.set_xticks([])
    sub.set_yticks([])
    sub.imshow(batch[i], interpolation='none')
