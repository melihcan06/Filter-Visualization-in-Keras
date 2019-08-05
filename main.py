# plot feature map of first conv layer for given image
"""from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from keras.models import load_model

def ozellik_haritasi():
    # load the model
    model = load_model("ResNet50")
    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[2].output)
    model.summary()
    # load the image with the required shape
    img = load_img('kedi.jpg', target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()

def filtreler():
    # cannot easily visualize filters lower down
    #load model
    model = load_model("ResNet50")
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[2].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D
from keras.models import Sequential

def birlestir(dizi):#dizi=[filtre1(np.array),filtre2,...](liste)
    return np.concatenate(dizi,axis=1)

filtre_sayisi=4
filtre_boyutu=(3,3)
input_shape=(200,200,1)
img=cv2.imread("kedi.jpg",0)
img=cv2.resize(img,(input_shape[0],input_shape[1]))
img=np.resize(img,(1,input_shape[0],input_shape[1],1))

model=Sequential()
model.add(Conv2D(filtre_sayisi,filtre_boyutu,activation='relu',input_shape=input_shape))

#filtreleri degistirme
#1
"""laplas_filtre1=np.array([[0,1,0],[1,-4,1],[0,1,0]]).reshape((3,3,1))
laplas_filtre2=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]).reshape((3,3,1))
laplas_filtre3=np.array([[1,1,1],[1,-8,1],[1,1,1]]).reshape((3,3,1))
laplas_filtre4=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((3,3,1))
laplas_biaslar=np.array([0.1,0.1,0.1,0.1]).reshape(4,)#[0,0,0,0]
a=[laplas_filtre1,laplas_filtre2,laplas_filtre3,laplas_filtre4]
laplas_filtreler=birlestir(a)

#biaslar=(np.ones((1,4))/10.0).reshape(1,)
biaslar=np.array([0,0,0,0]).reshape(1,-1)
model.layers[0].set_weights((laplas_filtreler,biaslar))"""

#conv bastirma
"""tahmin=model.predict(img)#tahmin channel first 1,198,198,4
for i in range(filtre_sayisi):
    plt.subplot(filtre_sayisi, int(filtre_sayisi/2),i+1)
    plt.imshow(tahmin[0,:,:,i])
    plt.gray()
    plt.axis('off')
plt.show()"""

#filtre bastirma
filters,biases=model.layers[0].get_weights()#filtreler channel last 3,3,1,4 #biases size (4,)
print filters,biases,filters[:,:,:,0]
print 1
"""for i in range(filtre_sayisi):
    plt.subplot(filtre_sayisi, int(filtre_sayisi/2), i + 1)
    plt.imshow(np.resize(filters[:,:,:,i],filtre_boyutu))
    plt.gray()
    plt.axis('off')
plt.show()"""
