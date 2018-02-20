import keras
import os
import numpy
import time
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
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
from sklearn.metrics import confusion_matrix, classification_report


# setting non interactive plot
plt.ioff()
# sign labels
sign_labels = [
 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

# model istantiation
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
v = str(7.6) # version
#datasets_and_tests/GUALANDI_DATASETS/
train_data_dir = '../../DatasetGualandi_v' + v + '/train'
validation_data_dir = '../../DatasetGualandi_v' + v + '/validation'
testing_data_dir = '../../DatasetGualandi_v' + v + '/test'

nb_train_samples = 1099
nb_validation_samples = 1099    
epochs = 1
batch_size = 64
weights_name = 'weights/BR_CNN_model_DatasetGualandi_v' + v + '_'+str(epochs)+'.h5'
directory = "DatasetGualandi_v" + v 


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# Bheda-Rapdour model
model = istantiate_model(input_shape)

sgd = SGD(lr=0.1, momentum=0.99, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])


# checkpoint

if not os.path.exists("results/"+directory):
    os.makedirs("results/"+directory)

# val acc
filepath="results/" + directory + "/weights-improvement-epoch_{epoch:02d}-val_acc_{val_acc:.2f}.hdf5"
checkpoint_val_acc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_val_acc]

# acc
filepath="results/" + directory + "/weights-improvement-epoch_{epoch:02d}-acc_{acc:.2f}.hdf5"
checkpoint_acc = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_acc]

# this is the augmentation configuration we will use for training
# V 1
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# V 2
#train_datagen = ImageDataGenerator(
#    rescale = 1)

# this is the augmentation configuration we will use for testing:
# only rescaling
# V 1
test_datagen = ImageDataGenerator(rescale=1. / 255)

# V 2
#test_datagen = ImageDataGenerator(
#    rescale = 1)

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

testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
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

keras.callbacks.EarlyStopping(monitor='acc',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

model.save_weights(weights_name)

#evaluate model
score = model.evaluate_generator(validation_generator, nb_validation_samples//batch_size, workers=12)
print("Results:")
print('Evaluate loss:', score[0])
print('Evaluate accuracy:', score[1])

#predict model
scores = model.predict_generator(testing_generator, nb_validation_samples//batch_size, workers=12)

correct = 0
for i, n in enumerate(testing_generator.filenames):
    #print("i: ", i)
    #print("n: ", n)
    #print("scores: ", sign_labels[numpy.argmax(scores[i])])
    #print("n[0]: ",n[0])
    if n[0] == sign_labels[numpy.argmax(scores[i])]:
        correct += 1
    
print("Correct:", correct, " Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])

# CONFUSION MATRIX
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size + 1)
y_pred = numpy.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')

print(classification_report(validation_generator.classes, y_pred, target_names=sign_labels))

#print('Test predict:', scores)
#for score in scores:
    #print(numpy.argmax(score))

#confusion matrix
#y_true = numpy.array([0] * 1000 + [1] * 1000 + )
#y_pred = scores > 0.5

#confusion_matrix(y_true, y_pred)

#save loss on txt
loss_history = history.history["loss"]
numpy_loss_history = numpy.array(loss_history)
numpy.savetxt("results/loss_history(img_scn_model)"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".txt", numpy_loss_history*100, delimiter=",")
max_loss = numpy.amin(numpy_loss_history)
print "MIN_LOSS: ", max_loss

#save val_loss on txt
val_loss_history = history.history["val_loss"]
numpy_val_loss_history = numpy.array(val_loss_history)
numpy.savetxt("results/val_loss_history(img_scn_model)"+time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".txt", numpy_val_loss_history*100, delimiter=",")
max_val_loss = numpy.amin(numpy_val_loss_history)
print "MIN_VAL_LOSS: ", max_val_loss

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

print "save maximums, max_loss, max_val_loss, max_acc, max_val_acc..."
maxs = numpy.array([max_loss, max_val_loss, max_acc, max_val_acc])
numpy.savetxt("results/maximums.txt", maxs, delimiter=",")



# plots

if not os.path.exists("results/plot_imgs/"+directory):
    os.makedirs("results/plot_imgs/"+directory)

# 1) val acc e train acc evolution during training for each version

#list all data in history
#print(history.history.keys())
print("Saving plots...")
# summarize history for accuracy
fig1 = plt.gcf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0, 1)
plt.xlim(0, 150)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()

#save img
fig1.savefig("results/plot_imgs/"+directory+"/MODEL_ACC_"+ directory + "_e" + str(epochs) +time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".png")

# 2) val loss e train loss evolution during training for each version

# summarize history for loss
fig2 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0, 1)
plt.xlim(0, 150)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()

#save img
fig2.savefig("results/plot_imgs/"+directory+"/MODEL_LOSS_"+ directory + "_e" + str(epochs) +time.strftime("%d-%m-%Y")+time.strftime("%H:%M:%S")+".png")



