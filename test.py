import tensorflow as tf
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV


dog = np.load('data/star.npy')
octopus = np.load('data/octopus.npy')
bee = np.load('data/bee.npy')
hedgehog = np.load('data/hedgehog.npy')
giraffe = np.load('data/giraffe.npy')

print(dog.shape)
print(octopus.shape)
print(bee.shape)
print(hedgehog.shape)
print(giraffe.shape)
print("-----")
# add a column with labels
dog = np.c_[dog, np.zeros(len(dog))]
octopus = np.c_[octopus, np.ones(len(octopus))]
bee = np.c_[bee, 2*np.ones(len(bee))]
hedgehog = np.c_[hedgehog, 3*np.ones(len(hedgehog))]
giraffe = np.c_[giraffe, 4*np.ones(len(giraffe))]

#store the label codes in a dictionary
label_dict = {0:'dog', 1:'octopus', 2:'bee', 3:'hedgehog', 4:'giraffe'}

print(dog.shape)
print(octopus.shape)
print(bee.shape)
print(hedgehog.shape)
print(giraffe.shape)





X = np.concatenate((dog[:5000,:-1], octopus[:5000,:-1], bee[:5000,:-1], hedgehog[:5000,:-1], giraffe[:5000,:-1]), axis=0).astype('float32') # all columns but the last
y = np.concatenate((dog[:5000,-1], octopus[:5000,-1], bee[:5000,-1], hedgehog[:5000,-1], giraffe[:5000,-1]), axis=0).astype('float32') # the last column
print(X.shape,y.shape)



X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.3,random_state=0)

y_train_cnn = tf.keras.utils.to_categorical(y_train)
y_test_cnn =  tf.keras.utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]



# # reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
print(X_train_cnn.shape)
print(y_train_cnn[9500])
# plt.imshow(X_train_cnn[9500].reshape(28,28), cmap='gray')
# plt.savefig("test.png")
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
print("a", X_test_cnn.shape)
def cnn_model0():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='tanh',kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3), activation='tanh',kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='tanh',kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.Dense(256, activation='tanh',kernel_initializer='random_uniform'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='random_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model



def cnn_model1():
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




np.random.seed(57897)
# build the model
model_cnn = cnn_model1()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=10, batch_size=400)
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)

print('Final CNN accuracy: ', scores[1])
print(scores[2])
print(scores[0])
model_json = model_cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_cnn.save_weights("model.h5")
print("Saved model to disk")

