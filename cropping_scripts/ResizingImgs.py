import cv2
import time
import os
import gc

#enable garbage collector
gc.enable()
print("gc is enabled: ", gc.isenabled())

#defining folder
folder = "../../../FotogualandiRinominate/FotoGualandi17-01_Evisa"


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
        print("to: ", '../../../FotogualandiResized/FotoGualandi17-01_Evisa_Resized' + dir + '/' + filename )
        cv2.imwrite('./../../../FotogualandiResized/FotoGualandi17-01_Evisa_Resized/' + dir + '/' + filename , image)
        image = []