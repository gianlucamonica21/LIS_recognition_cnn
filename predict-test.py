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

#sys.stdout = open('output.txt','a') #redirect print to file

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32" # Use GPU with theano

v = str(7.2) # weights version

weights_name = 'weights/BR_CNN_model_DatasetGualandi_v' + v + '_250.h5' # weights name

# sign labels
sign_labels = [
 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

folder = "../../TestSignerIndepent_v1" # folder of test imgs

img_width, img_height = 64, 64 # dimensions of our images

nb_testing_samples = 808 # number of testing samples

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

# for i in range(0,21):
#     print "INFO: saving in dir " + sign_labels[i]
#     # read img from directory
#     testing_generator = test_datagen.flow_from_directory(
#         folder+"/"+sign_labels[i],
#         target_size=(img_width, img_height),
#         batch_size=64,
#         class_mode='categorical',
#         shuffle=False,
#         save_to_dir="augmented-imgs-signer-independent-v1/"+sign_labels[i],
#         save_prefix=sign_labels[i])

# NUOVA VERSIONE CON SAVE PREFIX
# scorro cartelle lettere
for dir in os.listdir(folder):
    # scorro img 
    for filename in os.listdir(folder+'/'+dir):
        img = load_img(folder+'/'+dir+'/'+filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the directory

        testing_generator = test_datagen.flow(x, batch_size=1,
                                  save_to_dir='augmented-imgs-signer-independent-v1/', save_prefix=dir)
        i = 0
        for batch in testing_generator:
            i += 1
            if i > 2:
                break  # otherwise the generator would loop indefinitely    

# CONFUSION MATRIX
# Y_pred = model.predict_generator(testing_generator, nb_testing_samples // 64 + 1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('\nConfusion Matrix')
# cm = confusion_matrix(testing_generator.classes, y_pred)
# print(cm)
# print('\nClassification Report')

# print(classification_report(testing_generator.classes, y_pred, target_names=sign_labels))

#print(testing_generator.classes)
# Show confusion matrix in a separate window
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()

predict_utils.read_and_pred_from_folder("augmented-imgs-signer-independent-v1", img_width, img_height, model, sign_labels, weights_name)

