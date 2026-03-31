import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model import VGG16
from dataloader import load_data
import tensorflow as tf
from config import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':


    train_ds,val_ds,test_ds = load_data(batch_size=BATCH_SIZE)
    model = VGG16(num_classes=10)



    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/vgg16_best.weights.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',factor=0.5,patience=5,min_delta=1e-4,cooldown=2,min_lr=1e-6,verbose=1
    )
    

    model.compile(optimizer=tf.keras.optimizers.Adam(LR,weight_decay=WEIGHT_DECAY),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    model.fit(train_ds,epochs=EPOCHS,validation_data=val_ds,callbacks=[model_checkpoint_callback,lr_reduce])
