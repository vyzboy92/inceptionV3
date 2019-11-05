# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications import InceptionV3
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from time import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-3
BS = 10

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
# print(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (299, 299))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]

    if label == 'low':
        label = 0
    elif label == 'medium':
        label = 1
    elif label == 'severe':
        label = 2

    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
inceptionv3 = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights='imagenet')
for layer in inceptionv3.layers:
    layer.trainable = False

model = models.Sequential()
model.add(inceptionv3)
model.add(layers.AveragePooling2D((8, 8), border_mode='valid', name='avg_pool'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
filepath="model-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, period=10)

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1, callbacks=[tensorboard, checkpoint])


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on damage/ no damage")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
print('=====')
