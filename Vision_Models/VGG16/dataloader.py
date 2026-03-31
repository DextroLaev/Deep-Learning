import tensorflow as tf
from config import *

def load_data(batch_size=BATCH_SIZE,val_split=0.3):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.ZeroPadding2D(padding=4),
        tf.keras.layers.RandomCrop(32, 32),
        tf.keras.layers.RandomContrast(0.1),
    ])

    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train/255.0

    x_test = x_test/255.0

    val_size = int(len(x_train)*val_split)
    
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]

    x_train = x_train[val_size:]
    y_train = y_train[val_size:]


    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(BATCH_SIZE).map(lambda x, y: (data_augmentation(x, training=True), y),).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds,val_ds,test_ds

if __name__ == '__main__':
    load_data(batch_size=32)