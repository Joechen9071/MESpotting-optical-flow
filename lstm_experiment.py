from keras.engine.sequential import Sequential
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import LSTM, Activation, Dense
from keras import optimizers
from tensorflow.python.ops.array_ops import sequence_mask
import cv2

image 
n_step = 28
#每次输入的维度
n_input = 28
#分类类别数
n_classes = 10
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#将数据转为28*28的数据（n_samples,height,width）
x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#标准化数据，因为像素值在0-255之间，所以除以255后数值在0-1之间
x_train /= 255
x_test /= 255

#y_train，y_test 进行 one-hot-encoding，label为0-9的数字，所以一共10类
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)


model = Sequential()
learning_rate = 0.001

batch_size = 28

epochs = 2

model.add(LSTM(units=20,input_shape=(28,28)))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.summary()

model.compile(
    optimizer = optimizers.Adam(learning_rate = learning_rate),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

#训练
model.fit(x_train, y_train, 
          epochs = epochs, 
          batch_size= batch_size, 
          verbose = 1, 
          validation_data = (x_test,y_test))
#评估
score = model.evaluate(x_test, y_test, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss:',score[0])
print('acc:',score[1])
