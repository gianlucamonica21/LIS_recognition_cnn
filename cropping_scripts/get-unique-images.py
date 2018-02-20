import os

# Script per creare una cartella contenente una solo foto di ogni angolazione per candidato e per lettera
# Struttura della cartella creata:
# 	datasetUnique/
#		a/
#			a_ANTONIA_1_bottom.jpg
#			a_ANTONIA_12_top.jpg
#			a_ANTONIA_23_right.jpg
#			a_ANTONIA_56_left.jpg
#			a_ANTONIA_89_front.jpg
#			... (other candidates)
#		b/
#			...

INITIAL_FOLDER = "/home/maghi/Desktop/ml-datasets/DatasetGualandiUnito"
FINAL_FOLDER = "/home/maghi/Desktop/ml-datasets/DatasetGualandi-unique/"

NUM_CANDIDATI = 9
NUM_ANGOLAZIONI = 5

CANDIDATES_LIST = ["ANTONIA", "EVISA", "FATIMA", "GIANLUCA", "HUZAIFA", "IBRAHIM", "KENZA", "MARGHERITA", "MIKE"]
ANGLES_LIST = ["front", "top", "bottom", "left", "right"]

# Scorro le cartelle delle lettere
for dir in os.listdir(INITIAL_FOLDER):
	print("Letter: " + dir)
	photosCopied = 0 # contatore di foto copiate
	# Creo la cartella corrispondente nella cartella di destinazione
	os.makedirs(FINAL_FOLDER + "/" + dir)
	# Per ogni lettera, devo copiare solo UNA foto per candidato e per angolazione
	# Creo matrice NUM_CANDIDATIxNUM_ANGOLAZIONI con i campi inizializzati a FALSE (per memorizzare quali foto ho gia copiato)
	copiedPicturesMatrix = [[False for i in range(NUM_ANGOLAZIONI)] for j in range(NUM_CANDIDATI)]
	# Scorro le foto nella cartella
	for filename in os.listdir(INITIAL_FOLDER+"/"+dir):
		for candidateIndex, candidate in enumerate(CANDIDATES_LIST):
			for angleIndex, angle in enumerate(ANGLES_LIST):
				if candidate in filename and angle in filename:
					if copiedPicturesMatrix[candidateIndex][angleIndex] == False:
						# Non ho ancora una foto di questa angolatura per questo candidato
						# Copio la foto nel cartella di destinazione
						os.rename(INITIAL_FOLDER+"/"+dir+"/"+filename, FINAL_FOLDER+"/"+dir+"/"+filename)
						# Aggiorno contatore delle foto copiate
						photosCopied += 1
						# Setto il valore corrispondente della matrice a TRUE
						copiedPicturesMatrix[candidateIndex][angleIndex] = True
	print(str(photosCopied) + " pictures copied")
