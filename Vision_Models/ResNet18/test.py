import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys

from model import ResNet18,BasicBlock
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *

if __name__ == '__main__':
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model = ResNet18(num_classes=10)
    model(tf.random.normal([1,32,32,3]))
    
    model.load_weights('checkpoints/resnet18_best.weights.h5')
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print(model.evaluate(test_ds))
