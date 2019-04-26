import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras.layers import TimeDistributed,Conv2D,MaxPooling2D,Dropout,LSTM,Dense
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
import sys
from time import time
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt

def print_Cstyle_array(dir):
    print("{", end=" ")
    for filename in os.listdir(dir)[:totalclasses]:
        print("\"" + filename.replace(".npy", "") + "\",", end=" ")
    print("}", end=" ")


dir = "data/"
class_count = len(os.listdir(dir))
data = []
classcounter = 0
totalclasses = 100
classcounter = 0
samples = 15000
label_dict = {}
for filename in os.listdir(dir)[:totalclasses]:
    tmp = np.load(dir + filename)
    tmp = np.c_[tmp, classcounter * np.ones(len(tmp))]
    label_dict.update({filename.replace(".npy","") : classcounter})
    print("add to dict" , filename.replace(".npy",""), classcounter)
    data.append(tmp[:samples,:])
    classcounter += 1

data = np.asarray(data)
print("total shape" , data.shape)
print(label_dict)
X = data[:,:,:-1]
shape = X.shape[0]*X.shape[1]
X.shape = (shape, 784)
y = data[:totalclasses,:samples,-1]
y.shape = shape
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.4,random_state=100)
X_test, X_realtest, y_test, y_realtest = train_test_split(X_test,y_test,test_size=0.3,random_state=100)


y_train_cnn = tf.keras.utils.to_categorical(y_train)
y_test_cnn =  tf.keras.utils.to_categorical(y_test)
y_realtest_cnn = tf.keras.utils.to_categorical(y_realtest)
num_classes = y_test_cnn.shape[1]

X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')
X_realtest_cnn = X_realtest.reshape(X_realtest.shape[0], 28, 28,1).astype('float32')


def cnn_model3():
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1),activation = 'relu',kernel_initializer='RandomUniform'))
    BatchNormalization(axis=-1)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization(axis=-1)
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu',kernel_initializer='RandomUniform'))
    BatchNormalization(axis=-1)
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    BatchNormalization()
    model.add(tf.keras.layers.Dense(512,activation = 'relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(256,activation = 'relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(128, activation = 'relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='RandomUniform'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    tensorboard = TensorBoard(log_dir="logs/cnn_model3_deep_{}".format(time()),histogram_freq=1,
          write_graph = True, write_images = True,write_grads = True, batch_size = 100)
    return model,tensorboard

def cnn_model2():
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1),activation = 'tanh',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3), activation = 'tanh',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256,activation = 'tanh',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(128, activation = 'tanh',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='RandomUniform'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorycal_accuracy'])
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()),histogram_freq=1,
          write_graph = True, write_images = True,write_grads = True, batch_size = 100)
    return model,tensorboard



def cnn_model1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

def cnn_model0():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3), activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(128, activation='relu',kernel_initializer='RandomUniform'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='RandomUniform'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    tensorboard = TensorBoard(log_dir="logs/cnn_model0{}".format(time()),histogram_freq=1,
          write_graph = True, write_images = True,write_grads = True)
    return model,tensorboard


def cnn_model_lstm():
    model = tf.keras.Sequential()
    model.add(LSTM(128, activation='relu',input_shape=(28, 28), return_sequences=True))
    model.add(LSTM(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    tensorboard = TensorBoard(log_dir="logs/LSTM_{}".format(time()),histogram_freq=1,
          write_graph = True, write_images = True,write_grads = True, batch_size = 100)
    model.summary()
    return model,tensorboard


def cnn_model_leaky():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1),kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3),kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dense(256, kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='random_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model


np.random.seed(0)
# build the model
model_cnn,tensorboard = cnn_model3()
model_cnn.summary()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=10, batch_size=200, callbacks=[tensorboard])
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final Validation accuracy: ', scores[1])
scores = model_cnn.evaluate(X_realtest_cnn, y_realtest_cnn, verbose=0)
print('Final test accuracy: ', scores[1])


model_json = model_cnn.to_json()
with open("model_cnn_deep_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_cnn.save_weights("model_cnn_deep_2.h5")
print("Saved model to disk")

