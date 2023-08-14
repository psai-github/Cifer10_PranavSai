# import numpy as np
# from matplotlib import *
# import streamlit as st
# from PIL import Image
# import tensorflow
# from keras import cifar10,Sequential,to_categorical,Flatten,Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten,Dense

def create_cifer10_model():
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()
    X_train=X_train/255
    X_test=X_test/255
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)

    model = Sequential([
    Flatten(input_shape = (32,32,3)),
        Dense(1000,activation = 'relu'),
        Dense(10,activation = 'softmax')
    ])

    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
    model.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_test,y_test))
    model.save('cifar10_model.h5')
create_cifer10_model()