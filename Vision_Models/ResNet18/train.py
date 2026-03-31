import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys

from model import ResNet18
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

data_augmentation = tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=4),
        tf.keras.layers.RandomCrop(32,32),
        tf.keras.layers.RandomFlip('horizontal')
    ])

if __name__ == '__main__':

    

    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.map(
        lambda x,y: (data_augmentation(x,training=True),y),num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    

    model = ResNet18(num_classes=10)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/resnet18_best.weights.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1,mode='max'
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',factor=0.1,patience=10,verbose=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR,weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    
    history = model.fit(train_ds,epochs=EPOCHS,validation_data=test_ds,callbacks=[model_checkpoint_callback,lr_reduce])