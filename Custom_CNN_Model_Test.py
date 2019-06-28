# predict.py
import argparse
import sys
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model as load_keras_model
from keras.preprocessing.image import img_to_array, load_img

print(os.listdir("../input/flowers/flowers"))

# Any results you write to the current directory are saved as output.

x_ = list()
y = list()
IMG_SIZE = 256
for i in os.listdir("../input/flowers/flowers/daisy"):
    try:
        path = "../input/flowers/flowers/daisy/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        x_.append(img)
        y.append(0)
    except:
        None
for i in os.listdir("../input/flowers/flowers/dandelion"):
    try:
        path = "../input/flowers/flowers/dandelion/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        x_.append(img)
        y.append(1)
    except:
        None
for i in os.listdir("../input/flowers/flowers/rose"):
    try:
        path = "../input/flowers/flowers/rose/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        x_.append(img)
        y.append(2)
    except:
        None
for i in os.listdir("../input/flowers/flowers/sunflower"):
    try:
        path = "../input/flowers/flowers/sunflower/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        x_.append(img)
        y.append(3)
    except:
        None
for i in os.listdir("../input/flowers/flowers/tulip"):
    try:
        path = "../input/flowers/flowers/tulip/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        x_.append(img)
        y.append(4)
    except:
        None
x_ = np.array(x_)

# for replacement process i'll use keras.to_categorical 
from keras.utils.np_utils import to_categorical
y = to_categorical(y,num_classes = 5)

# test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_,y,test_size = 0.15,random_state = 42)

# validation and trains split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.15,random_state = 42)

# disable TF debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_filename = '../output/cusstom_cnn_model.h5'
class_to_name = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

def load_model():
    if os.path.exists(model_filename):
        return load_keras_model(model_filename)
    else:
        print("File {} not found!".format(model_filename))
        exit()


def load_image(filename):
    img_arr = img_to_array(load_img(filename))
    img_arr = cv2.resize(img_arr,(256,256))
    img_arr = np.reshape(img_arr,[256,256,3])
    return np.asarray([img_arr])


def predict(image, model):
    result = np.argmax(model.predict(image))
    return class_to_name[result]


if __name__ == '__main__':
    filename = '../input/515121050_dcb99890be.jpg'
    keras_model = load_model()
    
    from sklearn.metrics import confusion_matrix
    Y_pred = keras_model.predict(x_val)
    Y_pred_classes = np.argmax(Y_pred,axis = 1)
    Y_true = np.argmax(y_val,axis = 1)
    confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)
    f,ax = plt.subplots(figsize = (8,8))
    sns.heatmap(confusion_mtx,annot=True,linewidths = 0.01,cmap="Reds",
                linecolor = "gray",fmt = ".2f",ax=ax
                )
    plt.xlabel("Predicted label")
    plt.ylabel("True Label")
    plt.title("Confusion matrix")
    plt.show()
    
    print("Test Accuracy: {0:.2f}%".format(keras_model.evaluate(x_test,y_test)[1]*100))
    
    image = load_image(filename)
    image_class = predict(image, keras_model)

    originalImage = cv2.imread(filename)
    RGB_img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.title(image_class)
    plt.axis('off')
