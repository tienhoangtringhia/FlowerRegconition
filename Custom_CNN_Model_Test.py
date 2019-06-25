# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 

import seaborn as sns
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

img_width = 256
img_height = 256

if __name__ == "__main__":

    # load model
    model = keras.models.load_model('../output/cusstom_cnn_model.h5')
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # image path
    img_path = '../input/flowers/flowers/dandelion/510677438_73e4b91c95_m.jpg'

    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    img = cv2.resize(img,(256,256))
    img = np.reshape(img,[1,256,256,3])

    classes = model.predict_classes(img)

    print (classes)
