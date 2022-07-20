import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os


def move_file(filename, destination):
    def callback(event):
        os.rename(filename, destination + '/' + filename.split('/')[-1])
        plt.close()

    return callback


def rm_file(filename):
    def callback(event):
        os.remove(filename)
        plt.close()

    return callback


SRC_DIR = 'sample'
DEST_DIR = 'data'

# load images
images = []
image_names = []

for root, dirs, files in os.walk(SRC_DIR):
    for f in files:
        if f.split('.')[-1] in ('jpg', 'png'):
            images.append(plt.imread(root + '/' + f))
            image_names.append(root + '/' + f)

# display images
for i in range(len(images)):
    plt.title(image_names[i])

    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], interpolation='none')

    q_btn = Button(plt.axes([0.2, 0.1, 0.1, 0.05]), 'Q')
    q_btn.on_clicked(move_file(image_names[i],
                               DEST_DIR + '/q'))

    f_btn = Button(plt.axes([0.7, 0.1, 0.1, 0.05]), 'F')
    f_btn.on_clicked(move_file(image_names[i],
                               DEST_DIR + '/f'))

    b_btn = Button(plt.axes([0.45, 0.1, 0.1, 0.05]), 'B')
    b_btn.on_clicked(rm_file(image_names[i]))

    plt.show()
