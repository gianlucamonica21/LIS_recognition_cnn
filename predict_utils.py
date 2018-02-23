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
from keras.preprocessing.image import ImageDataGenerator

# predict a single img
def predict_img(img, img_width, img_height, model, weights_name):
    print("predicting...")
    model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])
    model.load_weights(weights_name)
 #   img = cv2.resize(img, (img_width, img_height)) 
 #   img = img/255.0
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img, batch_size= 1)
    #print pred
    classes = np.argmax(pred)
    return classes

# read and predict all imgs in folder
def read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels, weights_name):
    images = []
    total_imgs = 0
    correct_predictions = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        pred = predict_img(img, img_width, img_height, model, weights_name)
        print("Reading: ", filename, " --> prediction: ", sign_labels[pred])
        # check if prediction is correct
        if sign_labels[pred] == filename[0]:
            correct_predictions += 1
            print ("    RIGHT")
        # if prediction is wrong
        else:
            print ("    WRONG")
        total_imgs += 1    
        if total_imgs > 100:
            break

    print(correct_predictions, " correct predictions on", total_imgs , "total tests")
    print((correct_predictions * 100) / total_imgs, "%", "success on ", folder)