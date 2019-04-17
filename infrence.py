import tensorflow as tf
import  numpy as np
from matplotlib import pyplot as plt
tmp = np.load("data/rain.npy")
tmp = tmp[0]
tmp = tmp.reshape(1,28, 28,1).astype('float32')
print(tmp.shape)
# plt.imshow(tmp,cmap='gray')
# plt.savefig("test.png")

# Model reconstruction from JSON file
with open('model.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights('model.h5')
#print(model.summary())
print(model.predict(tmp))