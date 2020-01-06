import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

img = cv2.imread('C:/Users/Rahul/Desktop/data/Pikachu/b.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


################## Resizing all images ##################


desired_size = 368
im_pth = "C:/Users/Rahul/Desktop/data/Pikachu/a.jpg"

im = cv2.imread(im_pth)
old_size = im.shape[:2] # old_size is in (height, width) format

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])

# new_size should be in (width, height) format

im = cv2.resize(im, (new_size[1], new_size[0]))

delta_w = desired_size - new_size[1]
delta_h = desired_size - new_size[0]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)

color = [0, 0, 0]
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

cv2.imshow("image", new_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg', new_im) 


########### Reading multiple images at once

import glob

images = [cv2.imread(file) for file in glob.glob("C:/Users/Rahul/Desktop/data/Pikachu/*.jpg")]

images_1 = [cv2.imread(file) for file in glob.glob("C:/Users/Rahul/Desktop/data/Butterfree/*.jpg")]

images_2 = [cv2.imread(file) for file in glob.glob("C:/Users/Rahul/Desktop/data/Ditto/*.jpg")]



###############################################################

mera_dat = []

for i in range(199):
    desired_size = 368
    
    im = images[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat.append(new_im)


cv2.imshow("", mera_dat[120])


############# Butterfree

mera_dat_1 = []

for i in range(66):
    desired_size = 368
    
    im = images_1[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat_1.append(new_im)


########### Ditto
    

mera_dat_2 = []

for i in range(56):
    desired_size = 368
    
    im = images_2[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat_2.append(new_im)


arr = np.array(mera_dat)
arr = arr.reshape((199, 406272))

ar1 = np.array(mera_dat_1)
ar1 = ar1.reshape((66, 406272))

ar2 = np.array(mera_dat_2)
ar2 = ar2.reshape((56, 406272))

arr = arr / 255
ar1 = ar1 / 255
ar2 = ar2 / 255

dataset = pd.DataFrame(arr)
dataset['label'] = np.ones(199)

dataset.iloc[:, -1]

dataset_1 = pd.DataFrame(ar1)
dataset_1['label'] = np.zeros(66)

dataset_1.iloc[:, -1]

dataset_2 = pd.DataFrame(ar2)
dataset_2['label'] = np.array(np.ones(56) + np.ones(56))

dataset_2.iloc[:, -1]

dataset_master = pd.concat([dataset, dataset_1, dataset_2])

dataset_master.iloc[:, 406272]

X = dataset_master.iloc[:, 0:406272].values
y = dataset_master.iloc[:, -1].values


from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 3)
dtf.fit(X, y)

dtf.score(X, y)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X, y)

nb.score(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X, y)

svm.score(X, y)

from sklearn.cluster import KMeans

wcv = []

for i in range(1, 8):
    km = KMeans(n_clusters = i)
    km.fit(X)
    wcv.append(km.inertia_)

plt.plot(range(1, 8), wcv)
plt.show()


import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(256, activation = 'relu'))
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(3, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X, y, epochs = 5)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()























































































































