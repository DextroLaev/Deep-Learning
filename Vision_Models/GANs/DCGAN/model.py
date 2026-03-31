import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self,latent_dim=100,input_shape=7):
        super().__init__()
        self.dense = tf.keras.layers.Dense(input_shape*input_shape*256)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((input_shape,input_shape,256))
        
        self.conv_trans1 = tf.keras.layers.Conv2DTranspose(64,kernel_size=5,strides=2,padding='same')
        self.conv_trans2 = tf.keras.layers.Conv2DTranspose(32,kernel_size=5,strides=2,padding='same')
        self.conv_trans3 = tf.keras.layers.Conv2DTranspose(256,kernel_size=5,strides=2,padding='same')
        # self.conv_trans4 = tf.keras.layers.Conv2DTranspose(128,kernel_size=5,strides=2,padding='same')
        self.conv_trans5 = tf.keras.layers.Conv2DTranspose(1,kernel_size=5,strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
    
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
        x = self.conv_trans5(x)
        # x = self.bn3(x,training=training)
        # x = tf.nn.relu(x)
        # x = self.conv_trans4(x)
        # x = self.bn4(x,training=training)
        # x = tf.nn.relu(x)
        # x = self.conv_trans5(x)
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

        # self.conv3 = tf.keras.layers.Conv2D(256,kernel_size=5,strides=2,padding='same')
        # self.bn2 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        # self.fc1 = tf.keras.layers.Dense(32)
        # self.fc2 = tf.keras.layers.Dense(16)
        self.fc3 = tf.keras.layers.Dense(1)
    
    def call(self,x,training=False):
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x,0.2)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn1(x,training=training)
        x = tf.nn.leaky_relu(x,0.2)
        x = self.drop2(x)
        # x = self.conv3(x)
        # x = self.bn2(x,training=training)
        # x = tf.nn.leaky_relu(x,0.2)
        # x = self.dropout(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = tf.nn.leaky_relu(x,0.2)
        # x = self.fc2(x)
        # x = tf.nn.leaky_relu(x,0.2)
        x = self.fc3(x)
        return x