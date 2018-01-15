from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
root_dir = '\\tmp\\simpsons\\internet'

# sign labels
sign_labels = {'a': 0, 
'b': 1, 
'c': 2, 
'd': 3, 
'e': 4, 
'f': 5, 
'g': 6, 
'h': 7, 
'i': 8, 
'k': 9, 
'l': 10, 
'm': 11, 
'n': 12, 
'o': 13, 
'p': 14, 
'q': 15, 
'r': 16, 
's': 17, 
't': 18, 
'u': 19, 
'v': 20, 
'w': 21, 
'x': 22,
'y': 23}

rows = 3
cols = 3
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(7, 7))
fig.suptitle('Internet images', fontsize=20, y = 1.03)
count=0
for i in range(rows):
    for j in range(cols):
        all_files = os.listdir(root_dir)
        imgpath = os.path.join(root_dir, all_files[count])
        img = Image.open(imgpath)
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        ax[i][j].imshow(img)
        img = img_to_array(img)
        img = img/255.0
        img = img.reshape((1,) + img.shape)
        pred = model.predict(img, batch_size= 1)
        pred = pd.DataFrame(np.transpose(np.round(pred, decimals = 3)))
        pred = pred.nlargest(n = 3, columns = 0)
        pred['char'] = [list(chardict.keys())[list(chardict.values()).index(x)] for x in pred.index]
        charstr = ''
        for k in range(0,3):
            if k < 2:
                charstr = charstr+str(pred.iloc[k,1])+': '+str(pred.iloc[k,0])+'\n'
            else:
                charstr = charstr+str(pred.iloc[k,1])+': '+str(pred.iloc[k,0])
        ec = (0, .8, .1)
        fc = (0, .9, .2)
        count = count + 1
        ax[i][j].text(0, -10, charstr, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc, alpha = 0.7))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])