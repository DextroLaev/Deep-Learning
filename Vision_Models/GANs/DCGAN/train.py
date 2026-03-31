import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from config import *
from dataloader import *
from model import *
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    train_images,train_labels = load_data()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    generator_optim = tf.keras.optimizers.Adam(LR,beta_1=0.5)
    discriminator_optim = tf.keras.optimizers.Adam(LR,beta_1=0.5)

    dataset = tf.data.Dataset.from_tensor_slices(train_images.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=60000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
    fixed_noise = tf.random.normal((16, LATENT_DIM))
    gan_model = GAN(latent_dim=LATENT_DIM,input_shape=7)
    gan_model.compile(gen_optimizer=generator_optim,disc_optimizer=discriminator_optim)
    gan_model.train(epochs=EPOCHS,data=dataset,batch_size=BATCH_SIZE,fixed_noise=fixed_noise,K=1)


    
