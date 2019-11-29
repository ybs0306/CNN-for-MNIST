#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'kinoshitakenta'
__email__ = 'ybs0306748@gmail.com'


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
import dill


def label_map(ls, q):
    # according to the generated label map, corresponding labels
    for x, y in ls:
        if type(q) is str:
            if x is q:
                return y

        elif type(q) is int:
            if y is q:
                return x
    pass


def query_ans(q):
    # for answer questions
    ans = ''
    while ans is not 'y' and ans is not 'n':
        ans = input(q + " (y/n) : ").strip(' ')
        if ans is not 'y' and ans is not 'n':
            print('輸入格式錯誤 請重新輸入')

    if ans == 'y':
        return True
    else:
        return False


def import_data():

    dataset = []
    index = 0
    print("\n讀取檔案中 ...")

    # from data reading training data
    X_train = []
    y_train = []

    for dir_label in os.listdir('training'):
        if os.path.isdir('training/' + dir_label):
            dataset.append((dir_label, index))
            index += 1

            for files in os.listdir('training/' + dir_label):
                fullpath = 'training/' + dir_label + '/' + files
                image = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
                X_train.append(image)
                y_train.append(label_map(dataset, dir_label))

    # transform list to nparray
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    # reshape as : [samples][channels][rows][cols] by dim_ordering = 'th'
    X_train_np = X_train_np.reshape(
        X_train_np.shape[0], 1, 28, 28).astype('float32')

    # set range 0~255 to 0~1
    X_train_np = X_train_np / 255

    # let y's label output as one hot encode
    y_train_np = np_utils.to_categorical(y_train_np)
    num_classes = y_train_np.shape[1]

    print('\n#########################')
    print(f'train的資料量 : {X_train_np.shape[0]}')
    print(f'影像的高 : {X_train_np.shape[2]}')
    print(f'影像的寬 : {X_train_np.shape[3]}')
    print(f'label的資料種類(one hot encode處理過)) : {y_train_np.shape[1]}')
    print('#########################\n')

    f = open('dataset.pkl', 'wb')
    dill.dump(dataset, f)
    f.close()

    return X_train_np, y_train_np, num_classes


def build_model(X_train, y_train, num_classes):
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
    model.summary()

    # center_loss = get_center_loss(0.5, num_classes)

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=10, batch_size=200)

    # save model
    model.save_weights("Model_weight.h5")
    model.save("Model.h5")


def predict():
    f = open('dataset.pkl', 'rb')
    dataset = dill.load(f, "rb")
    f.close()
    # model.load_weights("Model_weight.h5")
    model = load_model("Model.h5")

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

            # delete extension png
            test_data_dict[dir_label[:-4]
                           ] = img.reshape(1, 1, 28, 28).astype('float32') / 255

    print(f'test data 共有{len(test_data_dict)}筆資料')

    # sort by image name
    test_data_dict_list = sorted(test_data_dict.keys())

    # output -> Answer.txt
    f = open('Answer.txt', 'w')
    for record in test_data_dict_list:
        predict = model.predict_classes(test_data_dict[record])
        # print(f'辨別為數字：{label_map(dataset, int(predict[0]))}')
        f.write(f'{record} {label_map(dataset, int(predict[0]))}\n')
    f.close()


def main():
    K.set_image_dim_ordering('th')
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    ################# training #################
    if query_ans('是否重新訓練模型 ?'):
        X_train, y_train, num_classes = import_data()
        build_model(X_train, y_train, num_classes)

    ################# testing ##################
    predict()


if __name__ == "__main__":
    main()
