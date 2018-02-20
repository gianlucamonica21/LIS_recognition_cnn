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

v = str(7.2)

weights_name = 'weights/BR_CNN_model_DatasetGualandi_v' + v + '_250.h5'
#weights_name = 'weights/BR_CNN_model_DatasetGualandi_v4.2.h5'

# check model weigths

cm_weights_name = 'results/weights-improvement-93-1.00.hdf5'

# creare matrice di predizione
a = np.zeros(shape=(22,22))

# letters_occ
let_occ = np.zeros(shape=(22))

# sign labels
sign_labels = [
 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

#folder = "../../datasets_and_tests/GUALANDI_DATASETS/DatasetGualandi_v" + v + "/testing"
#folder = "../../DatasetGualandi_v" + v + "/test"
folder = "../../TestSignerIndepent_v1"
# Use GPU with theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32"

# dimensions of our images.
img_width, img_height = 64, 64

#nb_validation_samples = 1099
#nb_validation_samples = 808
nb_validation_samples = 808
#nb_validation_samples = 867  

def predict_img(img, img_width, img_height, model):
    print("predicting...")
    model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])
    model.load_weights(weights_name)
    img = cv2.resize(img, (img_width, img_height)) 
    img = img/255.0
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img, batch_size= 1, verbose=1)
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
    print((correct_p * 100) / c, "%", "success on ", folder)
    print("front predicted: ", (f * 100)/ f_t, "%")
    print("top predicted: ", (t * 100)/ t_t, "%")
    print("bottom predicted: ", (b * 100)/ b_t, "%")
    print("left predicted: ", (l * 100)/ l_t, "%")
    print("right predicted: ", (r * 100)/ r_t, "%")
    #print("matrice di predizione")
    #for i in a:
    #    print ("indice del valore massimo ", i.max(), " :", sign_labels[np.argmax(i)])
    #for i in range(0,22):
        #print (sign_labels[i]," mispredicted as ", sign_labels[np.argmax(a[i,:])], " with ", np.max(a[i,:]))                 
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

model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])
model.load_weights(weights_name)

test_datagen = ImageDataGenerator(
    rescale=1. / 255)
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

testing_generator = test_datagen.flow_from_directory(
    folder,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
    save_to_dir="augmented-imgs-signer-independent-v1")


# CONFUSION MATRIX
Y_pred = model.predict_generator(testing_generator, nb_validation_samples // 64 + 1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(testing_generator.classes, y_pred)
print(cm)
print('Classification Report')

print(classification_report(testing_generator.classes, y_pred, target_names=sign_labels))

#print(testing_generator.classes)
# Show confusion matrix in a separate window
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()

#read_and_pred_from_folder(folder, img_width, img_height, model, sign_labels)

