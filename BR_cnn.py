'''This script is an adaptation to multiclass classification of the script from the blog post "Building powerful image classification 
models using very little data" from blog.keras.io.
Train samples: 52992
Validation samples: 12782
Number of classes: 24
'''
import keras
import os
import numpy
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import History
from keras.optimizers import Adam
import keras.regularizers as regularizers
import matplotlib.pyplot as plt
from hyperdash import Experiment
from keras.optimizers import SGD

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

# Use GPU with theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32"

# dimensions of our images.
img_width, img_height = 64, 64

#info
train_data_dir = '../../GUALANDI_DATASET_CROPPED'
validation_data_dir = '../../GUALANDI_DATASET_CROPPED'
nb_train_samples = 682
nb_validation_samples = 682
epochs = 30
batch_size = 64


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#Bheda-Rapdour model
model = istantiate_model(input_shape)

sgd = SGD(lr=0.1, momentum=0.99, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])

# checkpoint
filepath="results/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
#print train_generator.class_indices

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

model.save_weights('BR_CNN_model_GUALANDI.h5')

#evaluate model
score = model.evaluate_generator(validation_generator, nb_validation_samples//batch_size, workers=12)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#predict model
scores = model.predict_generator(validation_generator, nb_validation_samples//batch_size, workers=12)


#save loss on txt
loss_history = history.history["loss"]
numpy_loss_history = numpy.array(loss_history)
numpy.savetxt("results/loss_history(img_scn_model)"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".txt", numpy_loss_history*100, delimiter=",")
max_loss = numpy.amax(numpy_loss_history)
print "MAX_LOSS: ", max_loss

#save val_loss on txt
val_loss_history = history.history["val_loss"]
numpy_val_loss_history = numpy.array(val_loss_history)
numpy.savetxt("results/val_loss_history(img_scn_model)"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".txt", numpy_val_loss_history*100, delimiter=",")
max_val_loss = numpy.amax(numpy_val_loss_history)
print "MAX_VAL_LOSS: ", max_val_loss

#save acc on txt
acc_history = history.history["acc"]
numpy_acc_history = numpy.array(acc_history)
numpy.savetxt("results/acc_history(img_scn_model)"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".txt", numpy_acc_history*100, delimiter=",")
max_acc = numpy.amax(numpy_acc_history)
print "MAX_ACC: ", max_acc

#save val_acc on txt
val_acc_history = history.history["val_acc"]
numpy_val_acc_history = numpy.array(val_acc_history)
numpy.savetxt("results/val_acc_history(img_scn_model)"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".txt", numpy_val_acc_history*100, delimiter=",")
max_val_acc = numpy.amax(numpy_val_acc_history)
print "MAX_VAL_ACC: ", max_val_acc

print "save maximums, max_loss, max_val_loss, max_acc, max_val_acc"
maxs = numpy.array([max_loss, max_val_loss, max_acc, max_val_acc])
numpy.savetxt("results/maximums.txt", maxs, delimiter=",")

#create directories
directory="GUALANDI"

if not os.path.exists("results/plot_imgs/"+directory):
    os.makedirs("results/plot_imgs/"+directory)

#list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#save img
plt.savefig("results/plot_imgs/"+directory+"/MODEL_ACC"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".png")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#save img
plt.savefig("results/plot_imgs/"+directory+"/MODEL_LOSS"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".png")



