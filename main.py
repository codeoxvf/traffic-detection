import tensorflow as tf
from numpy import expand_dims
from matplotlib import pyplot as plt
from imread import imread_from_blob
from requests import get

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
saved_model = tf.keras.models.load_model('model')

model = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 256),
    saved_model
])

# run model
preds = [model.predict_on_batch(expand_dims(i, 0))[0] for i in batch]

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
    sub.set_xlabel('Prediction: '
                   + ('Free-flow' if preds[i][0] < 0 else 'Queue state')
                   + ' ' + str(preds[i]))

    sub.set_xticks([])
    sub.set_yticks([])
    sub.imshow(batch[i], interpolation='none')
