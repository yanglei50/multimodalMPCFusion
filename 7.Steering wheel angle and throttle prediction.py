# coding=utf-8
#https://blog.csdn.net/qq_24819773/article/details/90511453?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167006844516782414934230%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=167006844516782414934230&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-20-90511453-null-null.142^v67^control,201^v3^control_2,213^v2^t3_esquery_v3&utm_term=driving_log.csv&spm=1018.2226.3001.4187
import getopt
import sys

import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU, LeakyReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import backend as K
from keras.regularizers import l2
import os.path
from keras import Input, Model
from tensorflow.keras.utils import plot_model
import csv
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras import callbacks
import math
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

SEED = 13


def horizontal_flip(img, degree):
    '''
    按照50%的概率水平翻转图像
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)


def random_brightness(img, degree):
    '''
    随机调整输入图像的亮度， 调整强度于 0.1(变黑)和1(无变化)之间
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 调整亮度V: alpha * V
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)

    return (rgb, degree)


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    '''
    随机从左，中，右图像中选择一张图像， 并相应调整转动的角度
    img_address: 中间图像的的文件路径
    degree: 中间图像对于的方向盘转动角度
    degree_corr: 方向盘转动角度调整的值
    '''
    swap = np.random.choice(['L', 'R', 'C'])

    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(np.tan(degree) + degree_corr)
        return (img_address, corrected_label)
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(np.tan(degree) - degree_corr)
        return (img_address, corrected_label)
    else:
        return (img_address, degree)


def discard_zero_steering(degrees, rate):
    '''
    从角度为零的index中随机选择部分index返回
    degrees: 输入的角度值
    rate: 丢弃率， 如果rate=0.8， 意味着 80% 的index会被返回，用于丢弃
    '''
    steering_zero_idx = np.where(degrees == 0)
    steering_zero_idx = steering_zero_idx[0]
    size_del = int(len(steering_zero_idx) * rate)

    return np.random.choice(steering_zero_idx, size=size_del, replace=False)


def get_model(shape):
    '''
    预测方向盘角度: 以图像为输入, 预测方向盘的转动角度和油门throttle
    shape: 输入图像的尺寸, 例如(128, 128, 3)
    '''
    input_1 = Input(shape, name='input_1')
    # input_2 = Input(shape, name='input_2')
    conv_1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu')(input_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_1)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_2)
    maxpool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    conv_4 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_3)
    maxpool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_4)
    maxpool_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)

    conv_6 = Conv2D(512, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_5)
    maxpool_6 = MaxPooling2D(pool_size=(2, 2))(conv_6)

    conv_7 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_6)
    maxpool_7 = MaxPooling2D(pool_size=(2, 2))(conv_7)

    flat = Flatten()(maxpool_7)

    dense_1 = Dense(256, activation='relu')(flat)
    dense_2 = Dense(512, activation='relu')(dense_1)
    dense_3 = Dense(10, activation='relu')(dense_2)

    output_1 = Dense(1, activation='linear', name='output_1')(dense_3)
    output_2 = Dense(1, name='output_2')(dense_3)
    output_2_ = LeakyReLU(alpha=0.05)(output_2)

    model = Model(inputs=[input_1], outputs=[output_1, output_2_])
    sgd = Adam(lr=0.0001)
    # model.compile(optimizer=sgd, loss={'output_1':'mean_squared_error', 'output_2':'mean_squared_error'},
    #               loss_weights={'output_1': 1., 'output_2': 0.8})
    model.compile(optimizer=sgd, loss=['mean_squared_error', 'mean_squared_error'],
                  loss_weights=[1.0, 0.8])
    return model


# 图像数据增强
def image_transformation(img_address, degree, data_dir):
    img_address, degree = left_right_random_swap(img_address, degree)  # 图像的左右翻转
    img = cv2.imread(data_dir + img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, degree = random_brightness(img, degree)  # 图像亮度调整
    img, degree = horizontal_flip(img, degree)  # 图像水平翻转

    return (img, degree)


def generator_data(img, degree, throttle, batch_size, shape, data_dir='data/', discard_rate=0.65):
    y_bag = []
    z_bag = []
    x, y, z = shuffle(img, degree, throttle)
    rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
    new_x = np.delete(x, rand_zero_idx, axis=0)
    new_y = np.delete(y, rand_zero_idx, axis=0)
    new_z = np.delete(z, rand_zero_idx, axis=0)

    offset = 0
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))
        Z = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering, img_throttle = new_x[example + offset], new_y[example + offset], new_z[
                example + offset]
            img, img_steering = image_transformation(img_address, img_steering, data_dir)
            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5

            Y[example] = img_steering
            Z[example] = img_throttle
            y_bag.append(img_steering)
            z_bag.append(img_throttle)
            '''
             到达原来数据的结尾, 从头开始
            '''
            if (example + 1) + offset > len(new_y) - 1:
                x, y, z = shuffle(x, y, z)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_z = z
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                new_z = np.delete(new_z, rand_zero_idx, axis=0)
                offset = 0
        yield (X, [Y, Z])

        offset = offset + batch_size

        np.save('y_bag.npy', np.array(y_bag))
        np.save('z_bag.npy', np.array(z_bag))
        np.save('Xbatch_sample.npy', X)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "wdir=", ])
    for opt, arg in opts:
        if opt == '-h':
            print
            '8.riskhotmap.py -i <inputfile> -w <working directory>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            targetfile = arg
        elif opt in ("-w", "--wdir"):
            path = arg
            if not path.endswith('/'):
                path = path +'/'
    path = 'D:/DataContest/data/image2/mp/'
    targetfile = '1659666219.43_1659666263.64.csv'

    # data_path = 'D:/DataContest/data/image2/mp/'
    with open(path + 'carinfo.cvs', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)
    log = np.array(log)
    # 去掉文件第一行
    log = log[1:]

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    ls_imgs = glob.glob(path + 'scence*.png')
    # assert len(ls_imgs) == len(log) * 3, 'number of images does not match'
    avalid_data_length=min(len(ls_imgs),len(log) )
    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 32
    nb_epoch = 200
    x_=[]
    y_=[]
    z_=[]
    for i in range(0,avalid_data_length):
        x_.append(log[i][1])
        y_.append(log[i][2])
        z_.append(log[i][3])
    # x_ = log[:, 0]
    x_ = np.array(x_)
    y_ = np.array(y_)
    z_ = np.array(z_)
    # y_ = log[:, 3].astype(float)
    # z_ = log[:, 4].astype(float)
    print(x_.shape)
    print(y_.shape)
    print(z_.shape)
    x_, y_, z_ = shuffle(x_, y_, z_)
    X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(x_, y_, z_, test_size=validation_ratio,
                                                                      random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))

    samples_per_epoch = batch_size
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    model = get_model(shape)
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model_self.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')

    # 如果训练持续没有validation loss的提升, 提前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                                         verbose=0, mode='auto')
    callbacks_list = [early_stop, save_best]

    history = model.fit_generator(generator_data(X_train, y_train, z_train, batch_size, shape),
                                  steps_per_epoch=samples_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,
                                  validation_data=generator_data(X_val, y_val, z_val, batch_size, shape),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

    with open('./trainHistoryDict_self.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    # pyplot.title('model train vs validation loss')
    # pyplot.ylabel('loss')
    # pyplot.xlabel('epoch')
    # pyplot.legend(['train', 'validation'], loc='upper right')
    # pyplot.savefig('train_val_self.jpg')

    # 保存模型
    with open('model_self.json', 'w') as f:
        f.write(model.to_json())
    model.save('model_self.h5')
    print('Done!')
