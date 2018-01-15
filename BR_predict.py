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

def predict_img(img, img_width, img_height, model):
    print("predicting...")
    img = cv2.resize(img, (img_width, img_height)) 
    img = img/255.0
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img, batch_size= 1)
    classes = np.argmax(pred)
    return classes

def read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels):
    #Read the image with OpenCV
    images = []
    c = 0
    correct_p = 0
    for dir in os.listdir(folder):  
        print("dir: ", dir)
        n = 0
        for filename in os.listdir(folder+"/"+dir):
            print("reading: ", filename)
            img = cv2.imread(os.path.join(folder+"/"+dir,filename))
            pred = predict_img(img, img_width, img_height, model)
            print ("prediction: ", sign_labels[pred])
            c = c + 1
            if sign_labels[pred] == filename[0]:
                correct_p = correct_p + 1

    print(correct_p, " correct prediction on", c , "total tests")
    print((correct_p * 100) / c, "success %")
            #print("array img", images)

def istantiate_model(input_shape):
    model = Sequential()
    #1st
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #2st
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #3st
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #4st
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))

    #5st
    model.add(Dense(128))
    model.add(Dropout(0.5))

    #6st
    model.add(Dense(22))
    model.add(Activation('softmax'))

    model.summary()

    return model

#def predict_img(img):



# sign labels
sign_labels = [
 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

folder = "../../GUALANDI_DATASET_CROPPED"

# Use GPU with theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32"

# dimensions of our images.
img_width, img_height = 64, 64

#detecting input shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#instantiate model
model = istantiate_model(input_shape)

read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels)

# rows = 3
# cols = 3
# fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(7, 7))
# fig.suptitle('Internet images', fontsize=20, y = 1.03)
# count=0
# for i in range(rows):
#     for j in range(cols):
#         all_files = os.listdir(root_dir)
#         imgpath = os.path.join(root_dir, all_files[count])
#         img = Image.open(imgpath)
#         img = img.resize((img_width, img_height), Image.ANTIALIAS)
#         ax[i][j].imshow(img)
#         img = img_to_array(img)
#         img = img/255.0
#         img = img.reshape((1,) + img.shape)
#         pred = model.predict(img, batch_size= 1)
#         pred = pd.DataFrame(np.transpose(np.round(pred, decimals = 3)))
#         pred = pred.nlargest(n = 3, columns = 0)
#         pred['char'] = [list(chardict.keys())[list(chardict.values()).index(x)] for x in pred.index]
#         charstr = ''
#         for k in range(0,3):
#             if k < 2:
#                 charstr = charstr+str(pred.iloc[k,1])+': '+str(pred.iloc[k,0])+'\n'
#             else:
#                 charstr = charstr+str(pred.iloc[k,1])+': '+str(pred.iloc[k,0])
#         ec = (0, .8, .1)
#         fc = (0, .9, .2)
#         count = count + 1
#         ax[i][j].text(0, -10, charstr, size=10, rotation=0,
#                 ha="left", va="top", 
#                 bbox=dict(boxstyle="round", ec=ec, fc=fc, alpha = 0.7))
# plt.setp(ax, xticks=[], yticks=[])
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])