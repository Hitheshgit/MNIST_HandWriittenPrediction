# import the required libraries
from pyexpat import model
import cv2 as cv
from matplotlib import units
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# import mnist dataset
mnist = tf.keras.datasets.mnist

# split into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_train.shape

# normailze the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# fit the model
model.fit(x_train, y_train, epochs=1)

# printing the accuracy and loss
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)
# saving the model
model.save('digits.model')


# prediciton part
for x in range(1, 6):
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    # displaying the output
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
