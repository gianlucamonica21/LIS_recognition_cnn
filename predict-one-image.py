from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
import os
import cv2
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import BR_cnn_model # import BR_cnn model
import predict_utils # import predict-utils

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32" # Use GPU with theano

v = str(7.2) # weights version

weights_name = 'weights/BR_CNN_model_DatasetGualandi_v' + v + '_250.h5' # weights name

# sign labels
sign_labels = [
 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

image_path = "../../TestSignerIndepent_v1/b/b_GIULIA_30_front.JPG" # folder of test imgs

output_path = "../../TestSignerIndepent_v1_augmented/b_GIULIA_30_front{}.JPG"

img_width, img_height = 64, 64 # dimensions of our images

# detecting input shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# instantiate model
model = BR_cnn_model.istantiate_model(input_shape)
print "INFO: model instantiated"

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])
# load weights
model.load_weights(weights_name)
print "INFO: model compiled"

# augmentation of img
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


count = 10
# load image to array
image = img_to_array(load_img(image_path))

# reshape to array rank 4
image = image.reshape((1,) + image.shape)

# let's create infinite flow of images
images_flow = test_datagen.flow(image, batch_size=1)
for i, new_images in enumerate(images_flow):
    # we access only first image because of batch_size=1
    new_image = array_to_img(new_images[0], scale=True)
    new_image.save(output_path.format(i + 1))
    if i >= count:
		output_path_real = str(output_path.format(i + 1))
		print("output_path_real ", output_path_real)	
		break


# reading
print("reading ", output_path_real)
image_augmented = img_to_array(load_img(output_path_real))


# predict image
pred, percent = predict_utils.predict_img(image_augmented, img_width, img_height, model, weights_name)

print ("letta:", image_path,  "- predizione: ", sign_labels[pred], "con ", percent * 100, " %")