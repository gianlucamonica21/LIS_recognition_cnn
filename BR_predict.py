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

weights_name = 'BR_CNN_model_DatasetGualandi_v4.2.h5'

# creare matrice di predizione
a = np.zeros(shape=(22,22))

# array occorrenze per ogni lettera
array_letters = np.zeros(shape=(22))

# sign labels
sign_labels = [
 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

folder = "../../DatasetGualandi_v4.2/testing"

# Use GPU with theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32"

# dimensions of our images.
img_width, img_height = 64, 64

def predict_img(img, img_width, img_height, model):
    print("predicting...")
    model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])
    model.load_weights(weights_name)
    img = cv2.resize(img, (img_width, img_height)) 
    img = img/255.0
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img, batch_size= 1)
    #print pred
    classes = np.argmax(pred)
    percent = np.max(pred)
    return classes, percent

def read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels):
    #Read the image with OpenCV
    images = []
    c = 0
    # parameti predizione
    correct_p = 0
    f = 0
    l = 0
    r = 0
    t = 0
    b = 0
    f_t = 0
    l_t = 0
    r_t = 0
    t_t = 0
    b_t= 0
    for dir in os.listdir(folder):  
        print("dir: ", dir)
        n = 0
        for filename in os.listdir(folder+"/"+dir):
            #print("reading: ", filename)
            img = cv2.imread(os.path.join(folder+"/"+dir,filename))
            pred, percent = predict_img(img, img_width, img_height, model)
            c = c + 1
            if filename.find('front') != -1:
                f_t = f_t + 1
            elif filename.find('top') != -1:
                t_t = t_t + 1
            elif filename.find('left') != -1:
                l_t = l_t + 1
            elif filename.find('right') != -1:
                r_t = r_t + 1
            elif filename.find('bottom') != -1:
                b_t = b_t + 1  
            # se la predizione e corretta
            if sign_labels[pred] == filename[0]:
                print ("letter: ", filename[0], " --> prediction: ", sign_labels[pred], "OK with ", percent, " %")
                correct_p = correct_p + 1
                if filename.find('front') != -1:
                    f = f + 1
                elif filename.find('top') != -1:
                    t = t + 1
                elif filename.find('left') != -1:
                    l = l + 1
                elif filename.find('right') != -1:
                    r = r + 1
                elif filename.find('bottom') != -1:
                    b = b + 1
            # se la predizione non e corretta
            else:
                print ("letter: ", filename[0], " --> prediction: ", sign_labels[pred], "WRONG")
                a[ sign_labels.index(filename[0]), pred ] = a[ sign_labels.index(filename[0]), pred ] + 1


    print(correct_p, " correct prediction on", c , "total tests")
    print((correct_p * 100) / c, "%", "success ")
    print("front predicted: ", (f * 100)/ f_t, "%")
    print("top predicted: ", (t * 100)/ t_t, "%")
    print("bottom predicted: ", (b * 100)/ b_t, "%")
    print("left predicted: ", (l * 100)/ l_t, "%")
    print("right predicted: ", (r * 100)/ r_t, "%")
    #print("matrice di predizione")
    #for i in a:
    #    print ("indice del valore massimo ", i.max(), " :", sign_labels[np.argmax(i)])
    #for i in range(0,22):
        #print (sign_labels[i]," mispredicted as ", sign_labels[np.argmax(a[i,:])])                 
        #print ("success on ", sign_labels[i], " = ", np.sum(a[i,:]))



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