import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import shutil


#defining folder
folder = "../../../FotoGualandi24-01/FotoGualandi_Ibrahim"
letters = ['a', 'b', 'c']

# scorro cartelle delle lettere a, b, c, ...
for dir in os.listdir(folder):	
    print("dir: ", dir)
    n = 0
    # if dir not in letters:
    # scorro cartelle front, left, ...
    for subdir in os.listdir(folder+"/"+dir):
        print("subdir: ", subdir)
        # scorro file in front, left, ...
        for filename in os.listdir(folder+"/"+dir+"/"+subdir):
            print("filename:", filename)
            n = n + 1					
            file_to_rename = folder +"/" + dir + "/" + subdir + "/" + filename
            new_name = folder + "/" + dir + "/" + filename
            os.rename(file_to_rename, new_name)			
        # elimino cartella front, left, ecc
        shutil.rmtree(folder+"/"+dir+"/"+subdir)