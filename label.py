from get_images import get_images
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

# NOTE: make sure no duplicate initial letter
CLASSES = ['empty', 'light', 'heavy']

try:
    [os.makedirs(DEST_DIR + '/' + c[0]) for c in CLASSES]
except FileExistsError:
    pass

# load images
get_images()

images = []
image_names = []

for root, dirs, files in os.walk(SRC_DIR):
    for f in files:
        if f.split('.')[-1] in ('jpg', 'png'):
            images.append(plt.imread(root + '/' + f))
            image_names.append(root + '/' + f)

# display images
for i in range(len(images)):
    print(i, len(images))
    plt.title(image_names[i])

    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], interpolation='none')

    space = 0.8 / (len(CLASSES)+1)

    btns = []
    for j, cl in enumerate(CLASSES):
        btns.append(Button(plt.axes(
            [0.1 + space*j, 0.1, 0.1, 0.05]), cl))
        btns[-1].on_clicked(move_file(image_names[i], DEST_DIR + '/' + cl[0]))

    b_btn = Button(plt.axes([0.45, 0.2, 0.1, 0.05]), 'Remove')
    b_btn.on_clicked(rm_file(image_names[i]))

    plt.show()
