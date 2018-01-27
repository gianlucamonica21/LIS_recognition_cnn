import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os

#defining folder
folder = "../../../FotogualandiRinominate/FotoGualandi17-01_Evisa"

candidate_name = "EVISA"

for dir in os.listdir(folder):	
    print("dir: ", dir)
    for filename in os.listdir(folder+"/"+dir):
        print("filename: ", filename)		
        print("new filename: ", filename.replace("IBRAHIM", candidate_name, 1))
        file_to_rename = folder +"/" + dir + "/" + filename
        new_name = folder + "/" + dir + "/" + filename.replace("IBRAHIM", candidate_name, 1)
        print ("file to rename: ", file_to_rename)
        print ("new name: ", new_name)
        os.rename(file_to_rename, new_name)			
        