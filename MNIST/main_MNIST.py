#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'kinoshita kenta'
__email__ = 'ybs0306748@gmail.com'

import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras import backend as K
import sys
import os
import cv2
import numpy as np
import tensorflow as tf


def query_ans(q):
    # for answer question
    ans = ''
    while ans is not 'y' and ans is not 'n':
        ans = input(q + " (y/n) : ").strip(' ')
        if ans is not 'y' and ans is not 'n':
            print('輸入格式錯誤 請重新輸入')

    if ans == 'y':
        return True
    else:
        return False


def build_model(X_train, X_test, y_train, y_test, num_classes):
    # build model and train model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # center_loss = get_center_loss(0.5, num_classes)

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=200)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'CNN Error: {(100-scores[1]*100):.2f}%')
    # print("CNN Error: %.2f%%" % (100-scores[1]*100))

    model.save_weights("MNIST_weight.h5")
    model.save("MNIST_model.h5")
    # return model


def predict():
    # model.load_weights("MNIST_weight.h5")
    model = load_model("MNIST_model.h5")

    print('\n------------ 開始testing data ------------')
    if os.path.exists('testing'):
        pass
    else:
        print('測試資料夾不存在')
        sys.exit(0)

    # read data into dict
    test_data_dict = {}
    for dir_label in os.listdir('testing'):
        if dir_label != '.DS_Store':
            image = cv2.imread('testing/' + dir_label, cv2.IMREAD_GRAYSCALE)
            img = np.reshape(image, (1, 784)).astype('float32')

            # 去掉副檔名.png
            test_data_dict[dir_label[:-4]
                           ] = img.reshape(1, 1, 28, 28).astype('float32') / 255

    print(f'test data 共有{len(test_data_dict)}筆資料')

    # sort by image name
    test_data_dict_list = sorted(test_data_dict.keys())

    # output -> Answer.txt
    f = open('Answer.txt', 'w')
    for record in test_data_dict_list:
        predict = model.predict_classes(test_data_dict[record])
        # print (f'辨別為數字：{predict[0]}')

        f.write(f'{record} {predict[0]}\n')
    f.close()


def import_data():
    # fix dimension ordering issue
    # K.set_image_data_format('th')
    K.set_image_dim_ordering('th')
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape as : [samples][channels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # set range 0~255 to 0~1
    X_train = X_train / 255
    X_test = X_test / 255

    # let y's label output as one hot encode
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    print('\n#########################')
    print(f'train的資料量 : {X_train.shape[0]}')
    print(f'test的資料量 : {X_test.shape[0]}')
    print(f'影像的高 : {X_train.shape[1]}')
    print(f'影像的寬 : {X_train.shape[2]}')
    print(f'label的資料種類(one hot encode處理過)) : {y_test.shape[1]}')
    print('#########################\n')

    return X_train, X_test, y_train, y_test, num_classes


def main():
    if query_ans("要重新訓練MNIST資料集嗎 ?"):
        X_train, X_test, y_train, y_test, num_classes = import_data()
        build_model(X_train, X_test, y_train, y_test, num_classes)

    predict()


if __name__ == "__main__":
    main()
