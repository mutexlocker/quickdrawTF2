import tensorflow as tf
import  numpy as np
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt



tmp = Image.open('hat.png').convert('L')
tmp = PIL.ImageOps.invert(tmp)
tmp = np.asarray(tmp)/255.
print(tmp)

# tmp = np.load("data/star.npy")
# tmp = tmp[9]/255.

print(tmp)
print(tmp.shape)
plt.imshow(tmp.reshape(28,28),cmap='gray')
plt.savefig("test.png")
tmp = tmp.reshape(1,28, 28).astype('float32')


# Model reconstruction from JSON file
with open('model_cnn_lstm.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_cnn_lstm.h5')
print(model.summary())
np.set_printoptions(suppress=True)
tem_res = model.predict(tmp)
print(tem_res)
print(np.sort(tem_res))
print(np.argsort(tem_res))