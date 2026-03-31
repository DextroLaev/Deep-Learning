import tensorflow as tf
from config import *
import numpy as np

def sample_noise(batch_size,latent_shape):
    return np.random.randn(batch_size,latent_shape)

def sample_images(dataset,size):
    idx = np.random.choice(dataset.shape[0],size=size,replace=False)
    image_samples = dataset[idx]
    return image_samples