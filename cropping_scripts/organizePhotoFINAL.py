import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import shutil

# fasi:
# 1) rinominare foto in base a cartella "lettera" e cartelle "front","top", ecc
# 2) spostare le foto rinominate nelle cartella "lettera"
# 3) resizare tutte le foto all'interno delle cartelle "lettera"

INITIAL_FOLDER = "../../../FotoGualandi da organizzare/Nuova organizzazione/fase 0/Mike31-01"
FINAL_FOLDER = "../../../FotoGualandi da organizzare/Nuova organizzazione/fase 1/Mike31-01"
NAME = "MIKE"
    
# 1)
def rename(folder, name):
    print("____________________________________RENAMING___________________________________")
    # scorro cartelle "lettera"
    for dir in os.listdir(folder):	
        print("dir: ", dir)
        n = 0
        # scorro cartelle "posizione"
        for subdir in os.listdir(folder+"/"+dir):
            print("subdir: ", subdir)
            # scorro file "immagine"
            for filename in os.listdir(folder+"/"+dir+"/"+subdir):
                print("filename:", filename)
                n = n + 1			
                # rinomino		
                file_to_rename = folder +"/" + dir + "/" + subdir + "/" + filename
                new_name = folder + "/" + dir + "/" + subdir + "/" + dir + "_" + name + "_" + str(n) + "_" + subdir + "prova.JPG"
                os.rename(file_to_rename, new_name)

# 2)
def movePhotoByFolder(folder):
    print("____________________________________MOVING___________________________________")    
    # scorro cartelle "lettera"
    for dir in os.listdir(folder):	
        print("dir: ", dir)
        n = 0
        # scorro cartelle front, left, ...
        for subdir in os.listdir(folder+"/"+dir):
            print("subdir: ", subdir)
            # scorro file in front, left, ...
            for filename in os.listdir(folder+"/"+dir+"/"+subdir):
                print("filename:", filename)
                n = n + 1		
                # sposto file nella cartella superiore			
                file_to_rename = folder +"/" + dir + "/" + subdir + "/" + filename
                new_name = folder + "/" + dir + "/" + filename
                os.rename(file_to_rename, new_name)			
            # elimino cartella front, left, ecc
            shutil.rmtree(folder+"/"+dir+"/"+subdir)			

# 3)
def resize(folder, to_folder):
    print("____________________________________RESIZING___________________________________")    
    images = []
    c = 0
    # scorro cartella "lettera"
    for dir in os.listdir(folder):	
        print("dir: ", dir)
        # scorro file immagine        
        for filename in os.listdir(folder+"/"+dir):
            c = c + 1
            print("rimangono: ", len(os.listdir(folder)) - c)
            print("filename:", filename)
            #apro img
            image = cv2.imread(folder + "/" + dir + "/" + filename) 
            #resizing
            image = cv2.resize(image, (64,64))
            #salvo
            print("save to: ", to_folder + "/" + dir + '/' + filename )
            cv2.imwrite(to_folder + "/" + dir + '/' + filename , image)
            image = []      


# 1)
rename(INITIAL_FOLDER, NAME)
# 2)
movePhotoByFolder(INITIAL_FOLDER)
# 3)
resize(INITIAL_FOLDER, FINAL_FOLDER)