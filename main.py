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

#filtreleri degistirmeye çalış!

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
