import cv2
import numpy as np
import scipy
import random
import os
import pandas as pd
os.environ['KERAS_BACKEND'] = 'tensorflow'
import matplotlib.pyplot as plt
import sklearn
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from sklearn.externals import joblib

DATADIR = "/home/ajayrr/projects/ml/Parasite/Parasite/train/"
CATEGORIES = ["Parasitized", "Uninfected"]
IMG_SIZE = 80

training_data = []
training_label = []

# loading the data set
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to the directories
        class_num = CATEGORIES.index(category) # the two classes (encoding parasitized as [1,0] and [0,1])
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                training_data.append((img_array))
                if class_num == 0:
                    training_label.append(([1,0]))
                else:
                    training_label.append(([0,1]))
            except Exception as e:
                pass

create_training_data()

from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(training_data, training_label, test_size=0.2, random_state=0)

train_img = np.asarray(train_img)
test_img = np.asarray(test_img)
train_lbl = np.asarray(train_lbl)
test_lbl = np.asarray(test_lbl)

# normalization
train_img = train_img/255.0
test_img = test_img/255.0

model=Sequential()
model.add(Conv2D(filters=8,kernel_size=2,padding="same",activation="relu",input_shape=(80,80,3)))
model.add(Conv2D(filters=8,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu"))
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu"))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_img,train_lbl,validation_split=0.33,batch_size=50,epochs=10)
pickle.dump(model,open('cnn_model.sav','wb'))
#from keras.models import load_model
#model.save('model.h5')
print(model.evaluate(test_img, test_lbl))
pickle.dump(model,open('cnn_model.sav','wb'))
joblib.dump(model, 'cnn_model.pkl')
