import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os

#defining folder
folder = "../../../FotoGualandi24-01/FotoGualandi15-01_Huzaifa"

#renaming
for dir in os.listdir(folder):	
	print("dir: ", dir)
	n = 0
	for subdir in os.listdir(folder+"/"+dir):
		print("subdir: ", subdir)
		for filename in os.listdir(folder+"/"+dir+"/"+subdir):
			print("filename:", filename)
			n = n + 1					
			file_to_rename = folder +"/" + dir + "/" + subdir + "/" + filename
			new_name = folder + "/" + dir + "/" + subdir + "/" + dir + "_" + "IBRAHIM_" + str(n) + "_" + subdir + ".JPG"
			os.rename(file_to_rename, new_name)			
			#os.rename(folder +"/" + dir + "/" + subdir + "/" + filename, folder + "/" + dir + "/" + subdir + "/" + dir + "_" + "IBRAHIM_" + str(n) + "_" + subdir + ".JPG")
