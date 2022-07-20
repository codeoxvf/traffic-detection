import tensorflow as tf
from numpy import stack
from matplotlib import pyplot as plt
from matplotlib import image as img

# load images
image_names = ['data/day/4714.jpg', 'data/day/5799.jpg', 'data/day/2705.jpg',
               'data/day/1703.jpg']

# format images as tensor
arr = [tf.keras.utils.img_to_array(
    tf.keras.utils.load_img(i, target_size=(256, 256)))
    for i in image_names]

data = stack(arr)

# load saved model
model = tf.keras.models.load_model('model')

# run model
preds = model.predict_on_batch(data)

# display images
images = [img.imread(i) for i in image_names]

fig = plt.figure()

for i in range(len(image_names)):
    sub = fig.add_subplot(len(image_names)//2, 2, i+1)
    sub.set_title('Image ' + str(i) + ': ' + image_names[i].split('/')[-1])
    sub.set_xlabel('Prediction: '
                   + ('Free-flow' if preds[i][0] < 0 else 'Queue state')
                   + ' ' + str(preds[i]))

    sub.set_xticks([])
    sub.set_yticks([])
    sub.imshow(images[i], interpolation='none')

plt.show()
