import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 # pip install opencv-python
import os
import sys
from imp import reload
from PIL import Image #pip install image
import glob


# 1. data import
data = pd.read_csv('C:/Users/82104/Desktop/fer2013/fer2013.csv', header=0,index_col=None)
#data = np.loadtxt('C:/Users/82104/Desktop/fer2013/fer2013.csv',delimiter=',',dtype=np.str)
#data = np.genfromtxt('C:/Users/82104/Desktop/fer2013/fer2013.csv',delimiter=',',dtype=None)
print(data.head())

#####################################################################

# 2. data summary
## 2.1 data split
## 0 : Anger , 1 : Disgust , 2 : Fear , 3 : Happiness , 4 : Sadness , 5 : Surprise , 6 : Neutral
names = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happiness', 4:'Sadness', 5:'Surprise', 6:'Neutral'}

data_train = data.iloc[list(data['Usage']=='Training'),:] ; print(data_train.tail())
data_public_test = data.iloc[list(data['Usage']=='PublicTest'),:] ; print(data_public_test.tail())
data_private_test = data.iloc[list(data['Usage']=='PrivateTest'),:] ; print(data_private_test.tail())



## 2.2 the number of labels
### 2.2.1 total
y_total = data.iloc[:,0:1].values.flatten()
count_total = pd.value_counts(y_total, sort=True)
count_total = count_total.rename(index=names)
print(count_total) ; print("\nTotal:",sum(count_total))
### 2.2.2 train
y_train = data_train.iloc[:,0:1].values.flatten()
count_train = pd.value_counts(y_train, sort=True)
count_train = count_train.rename(index=names)
print(count_train) ; print("\nTrain:",sum(count_train))
### 2.2.3 private_test
y_private_test = data_private_test.iloc[:,0:1].values.flatten()
count_private_test = pd.value_counts(y_private_test, sort=True)
count_private_test = count_private_test.rename(index=names)
print(count_private_test) ; print("\nPrivate test:",sum(count_private_test))
### 2.2.4 public_test
y_public_test = data_public_test.iloc[:,0:1].values.flatten()
count_public_test = pd.value_counts(y_public_test, sort=True)
count_public_test = count_public_test.rename(index=names)
print(count_public_test) ; print("\nPublic test",sum(count_public_test))


## 2.3 pie plot
### 2.3.1 total
labels_total = count_total.index
sizes_total = count_total.values
fig1, ax1 = plt.subplots()
ax1.pie(sizes_total, labels=labels_total, autopct='%1.2f%%', shadow=True, startangle=90)
ax1.set_title('Total',fontsize=30)
### 2.3.2 train
labels_train = count_train.index
sizes_train = count_train.values
fig2, ax2 = plt.subplots()
ax2.pie(sizes_train, labels=labels_train, autopct='%1.2f%%', shadow=True, startangle=90)
ax2.set_title('Train',fontsize=30)
### 2.3.3 private_test
labels_private_test = count_private_test.index
sizes_private_test = count_private_test.values
fig3, ax3 = plt.subplots()
ax3.pie(sizes_private_test, labels=labels_private_test, autopct='%1.2f%%', shadow=True, startangle=90)
ax3.set_title('Private Test',fontsize=30)
### 2.3.4 public_test
labels_public_test = count_public_test.index
sizes_public_test = count_public_test.values
fig4, ax4 = plt.subplots()
ax4.pie(sizes_public_test, labels=labels_public_test, autopct='%1.2f%%', shadow=True, startangle=90)
ax4.set_title('Public Test',fontsize=30)
plt.show()

#####################################################################

# 3. data handling

# -*-coding: UTF-8 -*-
image_buffer =data['pixels']
images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])

X_train = images[list(data['Usage']=='Training'),:].reshape(28709,48,48)
X_private_test = images[list(data['Usage']=='PrivateTest'),:].reshape(3589,48,48)
X_public_test = images[list(data['Usage']=='PublicTest'),:].reshape(3589,48,48)

sum(images[list(data['Usage']=='Training'),:].sum(axis=1) == 0) # the number of black pictures in training data : 11
sum(images[list(data['Usage']=='PrivateTest'),:].sum(axis=1) == 0 ) # the number of black pictures in private test data : 0
sum(images[list(data['Usage']=='PublicTest'),:].sum(axis=1) == 0 ) # the number of black pictures in public test data : 1


# finally, you can check the real dataset you will use below.
data = data.drop(np.where(images.sum(axis=1)==0 )[0])
y = data['emotion']
X = np.delete(images, np.where(images.sum(axis=1)==0 )[0] , axis=0 ).reshape(-1,48,48)


X_train = X[list(data['Usage']=='Training'),:]
X_private_test = X[list(data['Usage']=='PrivateTest'),:]
X_public_test = X[list(data['Usage']=='PublicTest'),:]

y_train = y[list(data['Usage']=='Training')]
y_private_test = y[list(data['Usage']=='PrivateTest')]
y_public_test = y[list(data['Usage']=='PublicTest')]


# data save
os.getcwd()
os.chdir("C:\\Users\\82104\\Desktop\\fer2013\\mydata")
X_train_save = X_train.reshape(-1,48*48)
X_private_test_save = X_private_test.reshape(-1,48*48)
X_public_test_save = X_public_test.reshape(-1,48*48)

X_train_save = pd.DataFrame(X_train_save)
X_private_test_save = pd.DataFrame(X_private_test_save)
X_public_test_save = pd.DataFrame(X_public_test_save)

y_train_save = pd.DataFrame(y_train)
y_private_test_save = pd.DataFrame(y_private_test)
y_public_test_save = pd.DataFrame(y_public_test)


# save
X_train_save.to_csv("X_train.csv")
X_private_test_save.to_csv("X_private_test.csv")
X_public_test_save.to_csv("X_public_test.csv")
y_train_save.to_csv("y_train.csv")
y_private_test_save.to_csv("y_private_test.csv")
y_public_test_save.to_csv("y_public_test.csv")
#####################################################################

# 4. image showing & handling

## 4.1 image converting
usage = data[1:,2]
dataset = zip(images, usage) # the 'images' variable is in [3. data handling]
for i, d in enumerate(dataset):
    img = d[0].reshape((48,48))
    img_name = '%08d.jpg' %i
    img_path = os.path.join('C:/Users/82104/Desktop/fer2013/all_images', img_name)
    cv2.imwrite(img_path, img)


## 4.2 48 X 48 images with 5 examples
img1=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000001.jpg")
img2=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000300.jpg")
img3=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000003.jpg")
img4=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000008.jpg")
img5=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000004.jpg")
img6=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000016.jpg")
img7=cv2.imread("C:/Users/82104/Desktop/fer2013/all_images/00000012.jpg")

plt.subplot(1, 7, 1) ; plt.imshow(img1) ; plt.axis('off')
plt.subplot(1, 7, 2) ; plt.imshow(img2) ; plt.axis('off')
plt.subplot(1, 7, 3) ; plt.imshow(img3) ; plt.axis('off')
plt.subplot(1, 7, 4) ; plt.imshow(img4) ; plt.axis('off')
plt.subplot(1, 7, 5) ; plt.imshow(img5) ; plt.axis('off')
plt.subplot(1, 7, 6) ; plt.imshow(img6) ; plt.axis('off')
plt.subplot(1, 7, 7) ; plt.imshow(img7) ; plt.axis('off')

cv2.imshow('img1',img1) ; cv2.imshow('img2',img2) ; cv2.imshow('img3',img3) ; cv2.imshow('img4',img4)
cv2.imshow('img5',img5) ; cv2.imshow('img6',img6) ; cv2.imshow('img7',img7)



## 4.3 256 X 256 images with 5 examples
zoom1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
zoom2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_CUBIC)
zoom3 = cv2.resize(img3, (256, 256), interpolation=cv2.INTER_CUBIC)
zoom4 = cv2.resize(img4, (256, 256), interpolation=cv2.INTER_CUBIC)
zoom5 = cv2.resize(img5, (256, 256), interpolation=cv2.INTER_CUBIC)
zoom6 = cv2.resize(img6, (256, 256), interpolation=cv2.INTER_CUBIC)
zoom7 = cv2.resize(img7, (256, 256), interpolation=cv2.INTER_CUBIC)

cv2.imshow('Zoom1', zoom1) ; cv2.imshow('Zoom2', zoom2) ; cv2.imshow('Zoom3', zoom3) ; cv2.imshow('Zoom4', zoom4)
cv2.imshow('Zoom5', zoom5) ; cv2.imshow('Zoom6', zoom6) ; cv2.imshow('Zoom7', zoom7)

#####################################################################


