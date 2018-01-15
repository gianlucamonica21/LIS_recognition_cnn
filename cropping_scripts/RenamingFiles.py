import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os

#defining folder
# folder = "OriginalImg/ProveGualandi/4"
folder = "CroppedImg/ResultGualandi/dataset_recropped"


#Read the image with OpenCV
images = []
c = 0
for dir in os.listdir(folder):	
	print("dir: ", dir)
	n = 0
	for filename in os.listdir(folder+"/"+dir):
		n = n + 1
		print("filename:", filename)
		os.rename(folder +"/" + dir + "/" + filename, folder + "/" + dir + "/" + dir + "_" + "IBRAHIM_" + str(n) + ".JPG")
