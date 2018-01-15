import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import gc


#enable garbage collector
gc.enable()
print("gc is enabled: ", gc.isenabled())

base_val = [21, 3, 1]

# [opening, dilation, erosion]
alphabet_settings = {
    "a" : [50, 1, 1], #[45,20,1] per tutte, [50,20,1] per a_15, a_14, a_33, a_30 
    "b" : [50, 20, 1],
	#here
    "c" : [50, 1, 1],
    "d" : [21, 3, 1],
    "e" : [50, 20, 1],
    "f" : [50, 21, 1],
    "h" : [50, 1, 1],
    "i" : [70, 50, 1],
    "k" : [50, 1, 1],
    "l" : [100, 50, 1],
    "m" : [21, 3, 1],
    "n" : [50, 3, 1],
    "o" : [50, 3, 1],
    "p" : [65, 1, 1],
    "q" : [90, 1, 1],
    "r" : [65, 1, 1],
    "t" : [65, 1, 1],
    "u" : [75, 1, 1],
    "v" : [50, 1, 1],
    "w" : [25, 1, 1],
    "x" : [36, 1, 1],
    "y" : [36, 1, 1]
}

def CroppingImg(img, img_original, filename, dir):
	
	img_original = img

	# obtaining gray and thresholded img
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# removing noise
	gray = cv2.fastNlMeansDenoising(gray,None,10)
	ret, thresh = cv2.threshold(gray,12,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#thresh =  cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

	# noise removal
	kernel = np.ones((5,5),np.uint8)
	i_open = alphabet_settings[dir][0]
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = i_open)
	#closing
	# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations= 50 )
	# sure background area
	i_dilate = alphabet_settings[dir][1]
	sure_bg = cv2.dilate(opening, kernel,iterations = i_dilate)
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
	# erosion
	i_erode = alphabet_settings[dir][2]
	erosion = cv2.erode(opening,kernel,iterations = i_erode)
	ret, sure_fg = cv2.threshold(erosion,0.7*erosion.max(),255,0)
	# Finding unknown region
	sure_fg = np.uint8(sure_fg)

	#cv2.imwrite(os.path.join("CroppedImg", filename + "PROVA.jpg"), sure_bg) #Save the filtered image

	#2 sure foreground img -> cropped img

	#sure_bg = cv2.imread('CroppedImg/' + filename + 'PROVA.jpg')

	#img_to_crop = sure_bg

	#gray and edged version
	if sure_bg is not None:
		print("img processing...")
		#gray = cv2.cvtColor(sure_bg, cv2.COLOR_BGR2GRAY)
		gray = cv2.bilateralFilter(sure_bg, 11, 17, 17)
		edged = cv2.Canny(gray, 40, 255)
		#delete img prova
		# try: 
		# 	os.remove("CroppedImg/" + filename + "PROVA.jpg")
		# except: pass

	#dilation
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
	dilated = cv2.dilate(sure_bg, kernel)

	# find contours in the edged image, keep only the largest
	# ones, and initialize our screen contour
	(_, cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	# loop over our contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		
		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	mask = np.zeros_like(sure_bg)

	#img_mask = mask[cnts]


	#cv2.imwrite("CroppedImg/MASK" + filename, img_mask) #Save the filtered image


	height, width = edged.shape
	min_x, min_y = width, height
	max_x = max_y = 0

	#filling the circle if it exists
	cv2.drawContours(sure_bg, cnts, -1, (0, 0, 0), cv2.FILLED)
	edged = cv2.Canny(sure_bg, 50, 255)

	fg = cv2.bitwise_or(img, img, mask=cv2.bitwise_not(sure_fg))
	# cv2.imwrite("CroppedImg/" + filename + "BITWISED.jpg", fg)

	# mask = cv2.bitwise_not(sure_fg)
	# background = np.full(img.shape, 255, dtype=np.uint8)
	# bk = cv2.bitwise_or(background, background, mask=mask)

	# cv2.imwrite("CroppedImg/" + filename + "BITWISED2.jpg", bk)

	# final = cv2.bitwise_or(fg, bk)

	# cv2.imwrite("CroppedImg/" + filename + "FINAL.jpg", final)
	

	
	

	#finding contours of the img filled
	(_, cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	#drawing rectangle on the right contour
	c = max(cnts, key = cv2.contourArea)
	(x,y,w,h) = cv2.boundingRect(c)
	cv2.rectangle(sure_bg, (x,y), (x+w,y+h), (0, 255, 0), 2)

	#saving img with rectangle
	#cv2.imwrite("CroppedImg/RectGualandi/" + filename + "rect.jpg", sure_bg)

	#cropping original img
	img = fg [y: y + h, x: x + w]

	#directory = "CroppedImg/ResultGualandi/open" + str(i_open) + "-dilate" + str(i_dilate) + "-erode" + str(i_erode) + "__on_dataset_copia"
	directory = "CroppedImg/ResultGualandi/dataset_recropped"
	#creating directory and saving
	dir_letter = directory + "/" + dir
	if not os.path.exists(dir_letter):
		os.makedirs(dir_letter)
	cv2.imwrite(dir_letter + "/" + filename, img) #Save the filtered image



#1 read img -> sure foreground img

#reading img
# img = cv2.imread('OriginalImg/IMG_0383.JPG')
# if img is not None:
# 	print("loaded image")
# else:
# 	print("image not loaded")

# #reading and keeping original img
# img_original = cv2.imread('OriginalImg/IMG_0383.JPG')
# if img is not None:
# 	print("loaded image")
# else:
# 	print("image not loaded")

#defining folder
# folder = "OriginalImg/ProveGualandi/4"
folder = "../DatasetGualandi-black_copia"

#letter to modify
letters =  ["q_21.JPG", "q_19.JPG",]

#Read the image with OpenCV
images = []
c = 0
for dir in os.listdir(folder):	
	print("dir: ", dir)
	if dir == "q":
		print("using settings: ", dir, alphabet_settings[dir], " ------------------->")    		
		for filename in os.listdir(folder+"/"+dir):
			img = cv2.imread(os.path.join(folder+"/"+dir,filename))
			if filename in letters:
				print("processing: ", filename)
				if img is not None:
					c = c + 1
					print( len( letters ) - c," imgs left of", len( letters ))
					#cropping every image	
					CroppingImg(img,img,filename,dir)

