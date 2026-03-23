import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys

from model import ResNet18
import tensorflow as tf
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def train(model,epochs,train_ds,validation_ds):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x,y):
        with tf.GradientTape() as tape:
            logits = model(x,training=True)
            loss = loss_fn(y,logits)
        
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        preds = tf.argmax(logits,axis=1)
        y = tf.cast(tf.squeeze(y),tf.int64)
        correct = tf.reduce_sum(tf.cast(preds==y,tf.int32))
        return loss,correct

    @tf.function
    def val_step(x,y):
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        preds = tf.argmax(logits, axis=1)
        y = tf.cast(tf.squeeze(y),tf.int64)            
        correct = tf.reduce_sum(tf.cast(preds == y, tf.int32))
        return loss,correct


    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        print(f"\nEpoch {epoch+1}/{epochs}")

        for step,(x,y) in enumerate(train_ds):
            loss,correct = train_step(x,y)
            total_loss += loss.numpy()
            total_correct += correct
            total_samples += y.shape[0]

            sys.stdout.write(f'\rStep: {step+1}/{len(train_ds)} | loss: {loss.numpy():.4f} | acc: {correct/y.shape[0]:.4f}')
            sys.stdout.flush()
        
        print()

        train_loss = total_loss / len(train_ds)
        train_acc = total_correct / total_samples

        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        for x, y in validation_ds:
            loss,correct = val_step(x,y)
            val_loss += loss.numpy()
            val_correct += correct.numpy()
            val_samples += x.shape[0]

        val_loss /= len(validation_ds)
        val_acc = val_correct / val_samples

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

    return model

if __name__ == '__main__':
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128).prefetch(tf.data.AUTOTUNE)
    
    model = ResNet18(num_classes=10)
    train(model,10,train_ds,test_ds)
