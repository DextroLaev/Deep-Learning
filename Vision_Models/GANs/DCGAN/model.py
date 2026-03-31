import tensorflow as tf
from config import *
import sys
import matplotlib.pyplot as plt

class Generator(tf.keras.Model):
    def __init__(self,latent_dim=100,input_shape=7):
        super().__init__()
        self.dense = tf.keras.layers.Dense(input_shape*input_shape*256)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((input_shape,input_shape,256))
        
        self.conv_trans1 = tf.keras.layers.Conv2DTranspose(64,kernel_size=5,strides=2,padding='same')
        self.conv_trans2 = tf.keras.layers.Conv2DTranspose(32,kernel_size=5,strides=2,padding='same')        
        self.conv_trans3 = tf.keras.layers.Conv2DTranspose(1,kernel_size=5,strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        
    
    def call(self,x,training=False):
        x = self.dense(x)
        x = self.bn0(x,training=training)
        x = tf.nn.relu(x)
        x = self.reshape(x)
        x = self.conv_trans1(x)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv_trans2(x)
        x = self.bn2(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv_trans3(x)
        x = tf.nn.tanh(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(64,kernel_size=5,strides=2,padding='same')
        self.drop1 = tf.keras.layers.Dropout(0.3)

        self.conv2 = tf.keras.layers.Conv2D(128,kernel_size=5,strides=2,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()        
        self.fc3 = tf.keras.layers.Dense(1)
    
    def call(self,x,training=False):
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x,0.2)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn1(x,training=training)
        x = tf.nn.leaky_relu(x,0.2)
        x = self.drop2(x)
        x = self.flatten(x)        
        x = self.fc3(x)
        return x
    
class GAN(tf.keras.Model):
    def __init__(self,latent_dim=100,input_shape=7):
        self.generator = Generator(latent_dim=latent_dim,input_shape=input_shape)
        self.discriminator = Discriminator()
        self.latent_dim = latent_dim
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def compile(self,gen_optimizer,disc_optimizer):
        if gen_optimizer == None:
            gen_optimizer = tf.keras.optimizers.Adam(LR,beta_1=0.5)
        if disc_optimizer == None:
            disc_optimizer = tf.keras.optimizers.Adam(LR,beta_1=0.5)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
    
    @tf.function
    def discriminator_loss(self,real_output,fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output)*0.9,real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output),fake_output)
        return real_loss + fake_loss

    @tf.function
    def generator_loss(self,outputs):
        return self.loss_fn(tf.ones_like(outputs),outputs)
    
    @tf.function
    def train_discriminator(self,random_vector,image_samples):
        with tf.GradientTape() as tape:
            discriminator_real_output = self.discriminator(image_samples,training=True)
            generated_fake_image = self.generator(random_vector,training=True)
            discriminator_fake_output = self.discriminator(generated_fake_image,training=True)
            
            loss = self.discriminator_loss(discriminator_real_output,discriminator_fake_output)
        
        grads = tape.gradient(loss,self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads,self.discriminator.trainable_variables))
        return loss

    @tf.function
    def train_generator(self,random_vector):
        with tf.GradientTape() as tape:
            generated_fake_output = self.generator(random_vector,training=True)
            discriminator_output = self.discriminator(generated_fake_output,training=True)
            loss = self.generator_loss(discriminator_output)
        
        grads = tape.gradient(loss,self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads,self.generator.trainable_variables))
        return loss
    
    @tf.function
    def train_step(self,image_batch,batch_size,K=1):

        # Train the discriminator for K steps
        for k in range(K):
            random_vector_disc = tf.random.normal((batch_size,self.latent_dim))
            with tf.GradientTape() as tape:
                discriminator_real_output = self.discriminator(image_batch,training=True)
                generated_fake_image = self.generator(random_vector_disc,training=True)
                discriminator_fake_output = self.discriminator(generated_fake_image,training=True)
                
                disc_step_loss = self.discriminator_loss(discriminator_real_output,discriminator_fake_output)
            
            disc_grads = tape.gradient(disc_step_loss,self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(disc_grads,self.discriminator.trainable_variables))            
        
        # perform a single update on generator
        random_vector_gen = tf.random.normal((batch_size,self.latent_dim))
        with tf.GradientTape() as tape:
            generated_fake_output = self.generator(random_vector_gen,training=True)
            discriminator_output = self.discriminator(generated_fake_output,training=True)
            gen_step_loss = self.generator_loss(discriminator_output)
        
        grads = tape.gradient(gen_step_loss,self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads,self.generator.trainable_variables))
        
        return disc_step_loss,gen_step_loss  
    
    def save_generated_images(self,fixed_noise,path,epoch):
        images = self.generator(fixed_noise, training=False)
        images = (images + 1)/2
    
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i, :, :, 0].numpy(), cmap='gray')
            ax.axis('off')
        plt.suptitle(f'Epoch {epoch}', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{path}/epoch_{epoch:03d}.png', bbox_inches='tight')
        plt.close()
        
    
    def train(self,epochs,data,batch_size,fixed_noise,K=1,save_image_dir='./generated'):
        gen_plot = True if fixed_noise != None else False
        
        steps_per_epochs = 60000//batch_size
        if gen_plot:
            self.save_generated_images(fixed_noise,path=save_image_dir,epoch=0)

        for e in range(epochs):
            d_loss_avg = 0
            g_loss_avg = 0

            for step, image_batch in enumerate(data):
                disc_step_loss,gen_step_loss = self.train_step(image_batch,batch_size,K=K)

                d_loss_avg += disc_step_loss
                g_loss_avg += gen_step_loss

                if step % 200 == 0:
                    print(f'\r  Epoch {e+1}/{epochs} - step: {step+1}/{steps_per_epochs} -> Discriminator_loss: {disc_step_loss} | Generator_loss: {gen_step_loss}',end='',flush=True)

            d_loss_avg /= steps_per_epochs
            g_loss_avg /= steps_per_epochs
            print(f'\r  Epoch: {e+1} | Avg_Discriminator_loss: {d_loss_avg} | Avg_Generator_loss: {g_loss_avg}') 
            print()
            if gen_plot:
                self.save_generated_images(fixed_noise,path=save_image_dir,epoch=e+1)


