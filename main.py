import tensorflow as tf
from numpy import expand_dims, argmax
from matplotlib import pyplot as plt
from matplotlib import image as img

image1 = 'data/night/1004.jpg'
image2 = 'data/night/1704.jpg'

# load images
q = tf.keras.preprocessing.image.img_to_array(
    tf.keras.preprocessing.image.load_img(
        image1, target_size=(256, 256)
    )
)
f = tf.keras.preprocessing.image.img_to_array(
    tf.keras.preprocessing.image.load_img(
        image2, target_size=(256, 256)
    )
)
q = expand_dims(q, axis=0)
f = expand_dims(f, axis=0)

# load saved model
model = tf.keras.models.load_model('model')

#labels = ['b', 'f', 'i', 'q']
#print('Categorical labels:', labels)

# run model
pred1 = model.predict_on_batch(q)
pred2 = model.predict_on_batch(f)
print('Image 1', pred1)
print('Free-flow' if pred1[0][0] < 0 else 'Queue state')
#print('Label:', labels[int(argmax(pred1, axis=1)[0])])
print('Image 2', pred2)
print('Free-flow' if pred2[0][0] < 0 else 'Queue state')
#print('Label:', labels[int(argmax(pred2, axis=1)[0])])

# display images
img1 = img.imread(image1)
img2 = img.imread(image2)

fig = plt.figure()

sub1 = fig.add_subplot(2, 1, 1)
sub1.set_title('Image 1')
sub1.axis('off')
sub1.imshow(img1)

sub2 = fig.add_subplot(2, 1, 2)
sub2.imshow(img2)
sub2.set_title('Image 2')
sub2.axis('off')

plt.show()
