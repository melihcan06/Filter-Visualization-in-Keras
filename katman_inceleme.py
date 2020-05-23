from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D,Input
import numpy as np
import CNN.konvolusyon1 as knvl

model = Sequential()
model.add(Conv2D(filters=2, input_shape=(10,10,1), kernel_size=(3,3),strides=(1,1)))
model.add(Conv2D(filters=3, kernel_size=(3,3),strides=(1,1)))

model.summary()

np.random.seed(0)
x=np.random.random((1,10,10,1))

model2=Model(inputs=model.input,outputs=model.layers[0].output)

y_ilk=model2.predict(x)
y_son=model.predict(x)

girdi=x[0,:,:,0]
f0,b0=model.layers[0].get_weights()
f1,b1=model.layers[1].get_weights()

#birinci katman
f0_0_0=f0[:,:,0,0]
f0_0_1=f0[:,:,0,1]

knv=knvl.konvolusyon()
c_ilk_0=knv.konvolusyon_islemi(girdi,f0_0_0)
c_ilk_1=knv.konvolusyon_islemi(girdi,f0_0_1)

y_ilk_0=y_ilk[0,:,:,0]
y_ilk_1=y_ilk[0,:,:,1]

print(np.round(y_ilk_0-c_ilk_0,6))#0

#ikinci katman
f1_0_0=f1[:,:,0,0]
f1_1_0=f1[:,:,1,0]

f1_0_1=f1[:,:,0,1]
f1_1_1=f1[:,:,1,1]

f1_0_2=f1[:,:,0,2]
f1_1_2=f1[:,:,1,2]

y_son_0=y_son[0,:,:,0]
y_son_1=y_son[0,:,:,1]
y_son_2=y_son[0,:,:,2]

c_son_0_0=knv.konvolusyon_islemi(y_ilk_0,f1_0_0)
c_son_0_1=knv.konvolusyon_islemi(y_ilk_1,f1_1_0)

c_son_0=np.add(c_son_0_0,c_son_0_1)

print(np.round(y_son_0-c_son_0,6))#0
