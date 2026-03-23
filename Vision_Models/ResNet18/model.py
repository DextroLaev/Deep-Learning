
import tensorflow as tf

class BasicBlock(tf.keras.Model):
    def __init__(self,in_channels,out_channels,strides=1,**kwargs):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels,kernel_size=3,strides=strides,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,kernel_size=3,strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides!=1 or in_channels!=out_channels:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_channels,kernel_size=1,strides=strides),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.downsample = None
    
    def call(self,x,training=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out,training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out,training=training)
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = tf.nn.relu(out)
        return out

class ResNet18(tf.keras.Model):
    def __init__(self,num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 64
        self.initial_conv = tf.keras.layers.Conv2D(filters=64,strides=1,kernel_size=3,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(3,strides=2,padding='same')

        self.layer1 = self._make_layer(out_channels=64,blocks=2,strides=1)
        self.layer2 = self._make_layer(out_channels=128,blocks=2,strides=2)
        self.layer3 = self._make_layer(out_channels=256,blocks=2,strides=2)
        self.layer4 = self._make_layer(out_channels=512,blocks=2,strides=2)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)
    
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    
    @property
    def metrics(self):
        return [self.loss_tracker,self.accuracy]

    def _make_layer(self,out_channels,blocks,strides):
        layers = []
        layers.append(BasicBlock(self.in_channel,out_channels,strides=strides))
        self.in_channel = out_channels
        for _ in range(blocks-1):
            layers.append(BasicBlock(self.in_channel,out_channels,strides=1))
        return tf.keras.Sequential(layers)

    def call(self,x,training=False):
        x = self.initial_conv(x)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        # x = self.max_pool(x)

        for layer in [self.layer1,self.layer2,self.layer3,self.layer4]:
            x = layer(x,training=training)
        
        x = self.avg_pool(x)
        x = self.dropout(x,training=training)
        x = self.fc(x)
        return x
    
    @tf.function
    def train_step(self,data):
        x,y = data
        with tf.GradientTape() as tape:
            y_ = self(x,training=True)
            loss = self.compute_loss(y=y,y_pred=y_)

        grads = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))            
        
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y,y_)
        
        
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self,data):
        x,y = data
        y_ = self(x,training=False)
        loss = self.compute_loss(y=y,y_pred=y_)

        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y,y_)
        
        return {m.name: m.result() for m in self.metrics}
