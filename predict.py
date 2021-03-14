import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.keras")

print(model.predict(np.array([[0, 0]]))[0])
print(model.predict(np.array([[1, 0]]))[0])
print(model.predict(np.array([[0, 1]]))[0])
print(model.predict(np.array([[1, 1]]))[0])
