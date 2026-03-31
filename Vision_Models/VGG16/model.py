import tensorflow as tf
from config import *

class VGG16(tf.keras.Model):
    def __init__(self,num_classes):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv3 = tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv5 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')
        self.conv6 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')
        self.conv7 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.bn7 = tf.keras.layers.BatchNormalization()

        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv8 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
        self.conv9 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
        self.conv10 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.bn10 = tf.keras.layers.BatchNormalization()
        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv11 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
        self.conv12 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
        self.conv13 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
        self.bn11 = tf.keras.layers.BatchNormalization()
        self.bn12 = tf.keras.layers.BatchNormalization()
        self.bn13 = tf.keras.layers.BatchNormalization()
        self.max_pool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.ReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.fc1 = tf.keras.layers.Dense(512,activation='relu')
        self.fc2 = tf.keras.layers.Dense(512,activation='relu')
        self.fc3 = tf.keras.layers.Dense(num_classes)
        

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    @property
    def metrics(self):
        return [self.loss_tracker,self.accuracy]
    
    def call(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.max_pool1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.max_pool2(x)
        x = self.dropout1(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.max_pool3(x)
        x = self.dropout1(x)

        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))
        x = self.max_pool4(x)
        x = self.dropout1(x)

        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.relu(self.bn13(self.conv13(x)))
        x = self.max_pool5(x)
        x = self.dropout1(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
    
    @tf.function
    def train_step(self, train_data):
        x,y = train_data
        y = tf.cast(tf.squeeze(y,axis=-1),tf.int32)
        with tf.GradientTape() as tape:
            y_ = self(x,training=True)
        
            loss = self.compute_loss(y=y,y_pred = y_)
        grads = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y,y_)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self,test_data):
        x,y = test_data
        y_ = self(x,training=False)
        loss = self.compute_loss(y=y,y_pred = y_)
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y,y_)

        return {m.name:m.result() for m in self.metrics}
        
