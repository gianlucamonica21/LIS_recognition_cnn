import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import shutil


#defining from_folders
# from_folder = "../../../FotogualandiResized/FotoGualandi15-01_Huzaifa_Resized"
# from_folder = "../../../FotogualandiResized/FotoGualandi_Ibrahim_Resized"
# from_folder = "../../../FotogualandiResized/FotoGualandi17-01_Evisa_Resized"
from_folder = "../../../FotogualandiResized/FotoGualandi22-01_Kenza_Resized"

to_folder = "../../../DatasetGualandi_v1.2"

# scorro cartelle delle lettere a, b, c, ...
for dir_let in os.listdir(from_folder):	
    print("dir: ", dir_let)
    n = 0
    len_dir_let = len(os.listdir(from_folder+"/"+ dir_let))
    train_percent = (len_dir_let * 60) / 100
    validation_percent = (len_dir_let * 30) / 100
    testing_percent = len_dir_let - train_percent - validation_percent
    print("tot cartella ", dir_let, ": ", len_dir_let)
    print("train_percent: ", train_percent)
    print("validation_percent: ", validation_percent)
    print("testing_percent: ", testing_percent)
    # scorro imgs
    for filename in os.listdir(from_folder+"/"+ dir_let):
        print("filename:", filename)
        file_to_rename = from_folder +"/" + dir_let + "/" + filename        
        n = n + 1
        # training
        if n > 0 and n <= train_percent:
            new_name = to_folder + "/train/" + dir_let + "/" + filename
            os.rename(file_to_rename, new_name)
            print("copio ", n, "in training")
        elif n > train_percent and n <= (validation_percent + train_percent):
            new_name = to_folder + "/validation/" + dir_let + "/" + filename
            os.rename(file_to_rename, new_name)
            print("copio ", n, "in validation")            
        elif n > (validation_percent + train_percent):
            new_name = to_folder + "/testing/" + dir_let + "/" + filename
            os.rename(file_to_rename, new_name)
            print("copio ", n, "in testing")
            
