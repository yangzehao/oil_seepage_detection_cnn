# from keras.applications import VGG16
# from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import *
from keras.losses import *
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os



img_path = "seep_detection/train_images_256/"
mask_path = "seep_detection/train_masks_256/"

images = []
masks = []


def plot_image_mask(img, mask):
    fig, ax = plt.subplots(1, 2, figsize=(20, 16))

    ax[0].imshow(img)
    ax[0].set_title("Seep Image")

    ax[1].imshow(mask)
    ax[1].set_title("Mask")

    plt.show()


for filename in os.listdir(img_path):
    images.append(cv2.cvtColor(cv2.imread(img_path + filename), cv2.COLOR_BGR2RGB))
    masks.append(plt.imread(mask_path + filename))

images = np.array(images)
masks = np.array(masks)
id = 2
print(images.shape)
plot_image_mask(images[id], masks[id])

images_process = (images.astype('float32') / 255)

masks_process = (np.expand_dims(masks, axis=-1) > 0).astype('float32')
X_train, X_test, y_train, y_test = train_test_split(images_process, masks_process, test_size=0.8, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)


batch_size = 12

# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     fill_mode='constant',
                     cval=0, )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(
    X_train,
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

mask_generator = mask_datagen.flow(
    y_train,
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

from xception_hxt import *


def F1_score(y_true, y_pred, smooth=0.8):
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)

    mulO = k.sum(y_true_f * y_pred_f)
    return (2. * mulO + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)


def middle_loss(y_true, y_pred):
    smooth = 0.8
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    mulO = y_true_f * y_pred_f
    score = (2. * k.sum(mulO) + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)
    return 1. - score


def F1_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + middle_loss(y_true, y_pred)

# Iput shape for the images
model = xception((256, 256, 3))

#The F1_loss is used as the loss, and the F1_score can also be regarded as an accuracy metric
model.compile(optimizer=Adam(lr=0.001), loss=F1_loss, metrics=[F1_score, 'acc'])


history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    validation_steps=50,
    epochs=30)
