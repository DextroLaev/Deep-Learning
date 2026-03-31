import tensorflow as tf
from config import *

def load_data(batch_size=BATCH_SIZE):
    (train_images,train_labels),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    train_images = (train_images/127.5) - 1
    train_images = train_images.reshape(train_images.shape[0],28,28,1)
    return train_images,train_labels