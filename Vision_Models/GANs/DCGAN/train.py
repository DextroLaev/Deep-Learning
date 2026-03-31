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
from utils import sample_noise,sample_images
import sys

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
fixed_noise = tf.random.normal((16, LATENT_DIM))


def discriminator_loss(real_output,fake_output):
    # real_output = tf.nn.sigmoid(real_output)
    # fake_output = tf.nn.sigmoid(fake_output)
    # return -tf.reduce_mean(tf.math.log(real_output+1e-8) + tf.math.log(1-fake_output+1e-8))    
    real_loss = loss_fn(tf.ones_like(real_output)*0.9, real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(outputs):
    # outputs = tf.nn.sigmoid(outputs)
    # return -tf.reduce_mean(tf.math.log(outputs+1e-8))
    return loss_fn(tf.ones_like(outputs), outputs)

@tf.function
def train_generator(discriminator,generator,optimizer,noise_vectors):
    with tf.GradientTape() as tape:
        generator_output_images = generator(noise_vectors,training=True)
        discriminator_output = discriminator(generator_output_images,training=True)
        loss = generator_loss(discriminator_output)
    
    grads = tape.gradient(loss,generator.trainable_variables)   
    optimizer.apply_gradients(zip(grads,generator.trainable_variables))
    return loss

@tf.function
def train_discriminator(discriminator,generator,optimizer,noise_vectors,image_samples):
    with tf.GradientTape() as tape:
        discriminator_output_real = discriminator(image_samples,training=True)
        generated_output_images = generator(noise_vectors,training=True)
        discriminator_output_fake = discriminator(generated_output_images,training=True)

        loss = discriminator_loss(discriminator_output_real,discriminator_output_fake)
    
    grads = tape.gradient(loss,discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))
    return loss

def save_generated_images(generator, epoch):
    images = generator(fixed_noise, training=False)
    images = (images + 1)/2
 
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, :, :, 0].numpy(), cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}', y=1.01)
    plt.tight_layout()
    plt.savefig(f'generated/epoch_{epoch:03d}.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    train_images,train_labels = load_data()

    generator = Generator()  
    discriminator = Discriminator()  
    z1 = np.random.randn(32,100)

    generated = generator(z1)
    print(generated.shape)
    classify = discriminator(generated)
    print(classify.shape)


    generator_optim = tf.keras.optimizers.Adam(LR,beta_1=0.5)
    discriminator_optim = tf.keras.optimizers.Adam(LR,beta_1=0.5)

    dataset = tf.data.Dataset.from_tensor_slices(train_images.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=60000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = len(train_images) // BATCH_SIZE

    save_generated_images(generator,0)
    for e in range(EPOCHS):
        d_loss_avg = 0.0
        g_loss_avg = 0.0
        # noise_vectors = tf.random.normal((BATCH_SIZE, LATENT_DIM))

        for step, image_batch in enumerate(dataset):
            # disc_noise_vectors = tf.constant(sample_noise(batch_size=BATCH_SIZE,latent_shape=LATENT_DIM),dtype=tf.float32)
            disc_noise_vector = tf.random.normal((BATCH_SIZE,LATENT_DIM))
            gen_noise_vectors = tf.random.normal((BATCH_SIZE,LATENT_DIM))
            for k in range(K):
                # image_samples = tf.constant(sample_images(train_images,size=BATCH_SIZE),dtype=tf.float32)
                d_loss = train_discriminator(discriminator,generator,discriminator_optim,disc_noise_vector,image_batch)
            
            g_loss = train_generator(discriminator,generator,generator_optim,gen_noise_vectors)
            d_loss_avg += d_loss
            g_loss_avg += g_loss

            if step % 200 == 0:
                print(f'\r  Epoch: {e+1}/{EPOCHS} - Step: {step+1}/{steps_per_epoch} -> Discriminator_loss: {d_loss} | Generator_loss: {g_loss}', end='', flush=True)
        d_loss_avg /= steps_per_epoch
        g_loss_avg /= steps_per_epoch
        print(f'\n  Epoch: {e+1} | Avg_Discriminator_loss: {d_loss_avg} | Avg_Generator_loss: {g_loss_avg}')
        print()
        save_generated_images(generator,e+1)
            




    
