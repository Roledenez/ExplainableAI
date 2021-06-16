import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.4):
            print('\nloss is below 0.4, so cancelling the training')
            self.model.stop_training = True

callbacks = myCallback()

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels), (test_images,test_labels) = fashion_mnist.load_data()

plt.show(train_images[0])

print(train_labels[0])
print(train_images[0])

train_images = train_images/255.0
test_images = test_images/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(imput_shape=(28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy')

model.fit(train_images,train_labels,epochs=5)

model.evaluate(test_images,test_labels)

classification = model.predict(test_images)

print(classification[0])
print(test_labels[0])