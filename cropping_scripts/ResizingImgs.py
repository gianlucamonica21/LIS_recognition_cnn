import cv2
import time
import os
import gc

#enable garbage collector
gc.enable()
print("gc is enabled: ", gc.isenabled())

#defining folder
folder = "../DatasetNotCropped"


#Read the image with OpenCV
images = []
c = 0
for dir in os.listdir(folder):	
    print("dir: ", dir)
    for filename in os.listdir(folder+"/"+dir):
        print("filename:", filename)
        #opening img
        image = cv2.imread(folder + "/" + dir + "/" + filename) 
        #resizing
        image = cv2.resize(image, (64,64))
        #saving
        print("to: ", '../DatasetNotCropped_RESIZED/' + dir + '/' + filename )
        cv2.imwrite('../DatasetNotCropped_RESIZED/' + dir + '/' + filename , image)
        image = []