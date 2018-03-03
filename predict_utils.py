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
import BR_cnn_model # import BR_cnn model

# predict a single img
def predict_img(img, img_width, img_height, model, weights_name):
    img = img/255.0
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img)
    classes = np.argmax(pred)
    percent = np.max(pred)
    # print classes
    return classes, percent

# read and predict all imgs in folder
def read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels, weights_name):
    images = []
    total_imgs = 0
    correct_predictions = 0
    correct = False
    for filename in os.listdir(folder):
        correct = False
        img = cv2.imread(os.path.join(folder,filename))
        # cv2.imshow("cim",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pred = predict_img(img, img_width, img_height, model, weights_name)
        #print("Reading: ", filename, " --> prediction: ", sign_labels[pred])
        # check if prediction is correct
        if sign_labels[pred] == filename[0]:
            correct = True
            correct_predictions += 1
            #print ("---------------------RIGHT")
        # if prediction is wrong
        total_imgs += 1    
        if correct:
            print ("predicted ", total_imgs - 1, " img RIGHT")
        else:
            print ("predicted ", total_imgs - 1, " img ")
        if total_imgs > 808:
            break

    print(correct_predictions, " correct predictions on", total_imgs , "total tests")
    print((correct_predictions * 100) / total_imgs, "%", "success on ", folder)


# PROVE ---------------------------------------------------------------------------------------------------------------- 

#     def predict_img(img, img_width, img_height, model):
#     print("predicting...")
#     model.compile(loss='categorical_crossentropy',
#               optimizer='adam'  ,
#               metrics=['accuracy'])
#     model.load_weights(weights_name)
#     #img = cv2.resize(img, (img_width, img_height)) 
#     #img = img/255.0
#     img = img.reshape((1,) + img.shape)
#     pred = model.predict(img, batch_size= 1, verbose=1)
#     #print pred
#     classes = np.argmax(pred)
#     percent = np.max(pred)
#     return classes, percent

# def read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels):
#     #Read the image with OpenCV
#     images = []
#     c = 0
#     # parameti predizione
#     correct_p = 0
#     f = 0
#     l = 0
#     r = 0
#     t = 0
#     b = 0
#     f_t = 0
#     l_t = 0
#     r_t = 0
#     t_t = 0
#     b_t= 0
#     for dir in os.listdir(folder):  
#         print("dir: ", dir)
#         n = 0
#         for filename in os.listdir(folder+"/"+dir):
#             #print("reading: ", filename)
#             img = cv2.imread(os.path.join(folder+"/"+dir,filename))
#             pred, percent = predict_img(img, img_width, img_height, model)
#             c = c + 1
#             if filename.find('front') != -1:
#                 f_t = f_t + 1
#             elif filename.find('top') != -1:
#                 t_t = t_t + 1
#             elif filename.find('left') != -1:
#                 l_t = l_t + 1
#             elif filename.find('right') != -1:
#                 r_t = r_t + 1
#             elif filename.find('bottom') != -1:
#                 b_t = b_t + 1  
#             # se la predizione e corretta
#             if sign_labels[pred] == filename[0]:
#                 print ("letter: ", filename[0], " --> prediction: ", sign_labels[pred], "OK with ", percent, " %")
#                 correct_p = correct_p + 1
#                 if filename.find('front') != -1:
#                     f = f + 1
#                 elif filename.find('top') != -1:
#                     t = t + 1
#                 elif filename.find('left') != -1:
#                     l = l + 1
#                 elif filename.find('right') != -1:
#                     r = r + 1
#                 elif filename.find('bottom') != -1:
#                     b = b + 1
#             # se la predizione non e corretta
#             else:
#                 print ("letter: ", filename[0], " --> prediction: ", sign_labels[pred], "WRONG")
#                 a[ sign_labels.index(filename[0]), pred ] = a[ sign_labels.index(filename[0]), pred ] + 1


#     print(correct_p, " correct prediction on", c , "total tests")
#     print((correct_p * 100) / c, "%", "success on ", folder)
#     print("front predicted: ", (f * 100)/ f_t, "%")
#     print("top predicted: ", (t * 100)/ t_t, "%")
#     print("bottom predicted: ", (b * 100)/ b_t, "%")
#     print("left predicted: ", (l * 100)/ l_t, "%")
#     print("right predicted: ", (r * 100)/ r_t, "%")
#     #print("matrice di predizione")
#     #for i in a:
#     #    print ("indice del valore massimo ", i.max(), " :", sign_labels[np.argmax(i)])
#     #for i in range(0,22):
#         #print (sign_labels[i]," mispredicted as ", sign_labels[np.argmax(a[i,:])], " with ", np.max(a[i,:]))                 
#         #print ("success on ", sign_labels[i], " = ", np.sum(a[i,:]))