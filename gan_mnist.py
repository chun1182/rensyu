# -*- coding: utf-8 -*-
from __future__ import print_function, division
#gan
import os
import tensorflow as tf
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config =tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
K.set_session(sess)
import numpy as np
np.random.seed(777)

from keras.datasets import mnist
from keras.models import load_model, Sequential, Model
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout, Input, Activation, AveragePooling2D, UpSampling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adagrad, Adam
import matplotlib.pyplot as plt
import sys
import numpy as np

class GAN():
    def __init__(self):
        #mnistデータ用の入力データサイズ
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # 潜在変数の次元数 
        self.z_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # discriminatorモデル
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Generatorモデル
        self.generator = self.build_generator()
        # generatorは単体で学習しないのでコンパイルは必要ない
        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (self.z_dim,)
        model = Sequential()
        model.add(Dense(128*7*7, activation='relu', input_shape=noise_shape))
        model.add(Reshape((7,7,128)))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, (3,3),padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(UpSampling2D())
        model.add(Conv2D(64, (3,3),padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1, (3,3),padding='same', activation='tanh'))
        model.summary()

        return model

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()
        model.add(Conv2D(32, (3,3),padding='same',input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128, (3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model
    
    def build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def build_combined2(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()
        return model



    def train(self, epochs, batch_size=128, save_interval=50):

        # mnistデータの読み込み
        (X_train, _), (_, _) = mnist.load_data()

        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)
        
        for epoch in range(epochs):

            for iteration in range(num_batches):

                # ---------------------
                #  Discriminatorの学習
                # ---------------------

                # バッチサイズの半数をGeneratorから生成
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)


                # バッチサイズの半数を教師データからピックアップ
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                # discriminatorを学習
                # 本物データと偽物データは別々に学習させる
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                # それぞれの損失関数を平均
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                # ---------------------
                #  Generatorの学習
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.z_dim))

                # 生成データの正解ラベルは本物（1） 
                valid_y = np.array([1] * batch_size)

                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)

                # 進捗の表示
                print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" 
                       % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))

                # 指定した間隔で生成画像を保存
                if (iteration + epoch * num_batches) % save_interval == 0:
                    self.save_imgs(iteration + epoch * num_batches)

    def save_imgs(self, epoch):
        # 生成画像を敷き詰めるときの行数、列数
        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # 生成画像を0-1に再スケール
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()
        
        
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10, batch_size=128, save_interval=50)