import numpy as np
import utils
from keras.layers import Input
from keras.layers.core import Dense, Reshape, Lambda
from keras.layers.convolutional import Conv1D, Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import time
from stl10_input import read_all_images
#import skimage.io

#img = skimage.io.imread('test.jpg')
    
def yuv2rgb_kernel(shape, dtype=None):
    print(shape)
    return K.reshape(utils.rgb_from_yuv_T, shape)

yuv2rgbLayer = Conv2D(3, (1, 1), trainable=False, use_bias=False,
                      kernel_initializer=yuv2rgb_kernel)

def get_test(img_h=utils.IMG_H):
    inputs = Input(shape=(img_h, img_h, 3))
    x = yuv2rgbLayer(inputs)
    return Model(inputs=inputs, outputs=x)

def get_gen(img_h=utils.IMG_H):
    inputs = Input(shape=(img_h, img_h, 1))
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(inputs)
    concat1 = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(concat1)
    x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
    sum1 = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(sum1)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='elu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='elu', padding='same')(x)
    sum2 = Conv2D(256, (3, 3), activation='elu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(sum2)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='elu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='elu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='elu', padding='same')(x)
    concat4 = Conv2D(256, (1, 1), activation='elu', padding='same')(x)
    x = Add()([sum2, UpSampling2D()(concat4)]) #
    concat3 = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
    x = Add()([sum1, UpSampling2D()(concat3)]) #
    concat2 = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    #x = Concatenate()([concat1, UpSampling2D()(concat2)])
    x = Add()([concat1, UpSampling2D()(concat2)])
    #x = Concatenate()([concat1, UpSampling2D()(concat2), UpSampling2D(4)(concat3), UpSampling2D(8)(concat4)])
    x = Conv2D(2, (3, 3), activation='elu', padding='same')(x)
    out = Concatenate()([inputs, x])
    out = Conv2D(3, (1, 1), activation='tanh', padding='same')(out)
    #out_u = Conv2D(1, (3, 3), activation=tanh_u, padding="same")(x)
    #out_v = Conv2D(1, (3, 3), activation=tanh_v, padding="same")(x)
    #out = yuv2rgbLayer(out)
    out = Conv2D(3, (1, 1), activation="sigmoid")(out)
    return Model(inputs=inputs, outputs=out)

def get_disc(img_h=utils.IMG_H):
    ins = Input(shape=(img_h, img_h, 3))
    x = Conv2D(64, (3, 3), padding='same')(ins)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (3, 3), padding='same', activation="tanh", use_bias=False)(x)
    outs = GlobalAveragePooling2D()(x)
    return Model(inputs=ins, outputs=outs)
    
def get_gan(gen, disc):
    ins = Input(shape=gen.input_shape[1:])
    generated = gen(ins)
    outs = disc(generated)
    return Model(inputs=ins, outputs=outs)
    
def model_info(model, filename='model.png'):
    print(model.summary())
    plot_model(model, to_file=filename, show_shapes=True)

def train(fname, epochs=1, batch_size=50, to_load=None):
    images = read_all_images(fname)
    
    # get models and compile
    gen = get_gen()
    gen.compile(loss='mean_squared_error', optimizer='nadam')

    disc = get_disc()
    disc.trainable = True
    disc.compile(loss=utils.wasserstein, optimizer='nadam')
    disc._make_train_function()
    
    if to_load:
        if 'gen' in to_load:
            gen = utils.load(to_load['gen'])
            print("loaded generator")
        if 'disc' in to_load:
            disc = utils.load(to_load['disc'])
            print("loaded discriminator")
    
    disc.trainable = False
    GAN = get_gan(gen, disc)
    print(GAN.summary())
    GAN.compile(loss=utils.wasserstein, optimizer='nadam')
    ones = np.ones(batch_size)
    mones = -ones
    mixones = np.concatenate((mones, ones))

    # test save / load
    GAN.save('./models/GAN-test.h5')
    utils.load('./models/GAN-test')
    
    for j in range(epochs):
        np.random.shuffle(images)
        for i in range(0, len(images), batch_size):
            x = i + batch_size
            print('batch %d:' % (i//batch_size), end=' ', flush=True)
            start_time = time.time()
            batch_color = images[i:x]/255
            batch_grey = utils.to_grayscale(batch_color)
            batch_gen = gen.predict(batch_grey, batch_size=batch_size)
            disc.trainable = True
            for k in range(2):
                dloss = disc.train_on_batch(np.concatenate((batch_color, batch_gen)), mixones)
            disc.trainable = False
            for k in range(3):
                gloss = GAN.train_on_batch(batch_grey, mones)
            #print(GAN.evaluate(batch_grey, ones, verbose=0))
            print('dloss={:.4f}, gloss={:.4f}, time:{:.2f}'.format(dloss, gloss, time.time()-start_time))
            if ((x <= 10000 and x % 2500 == 0) or x % 5000 == 0):
                gen.save('./models/gen%d-%d.h5' % (j, x))
                disc.save('./models/disc%d-%d.h5' % (j, x))
                #GAN.save('./models/GAN%d-%d.h5' % (j, x))
        print()

def train_gen(fname, epochs=3, batch_size=50, to_load=None):
    if to_load:
        gen = utils.load('./regression models/gen' + to_load)
    else:
        gen = get_gen()
        gen.compile(loss='mean_squared_error', optimizer="nadam")
    images = read_all_images(fname)
    
    for j in range(epochs):
        np.random.shuffle(images)
        for i in range(0, len(images), batch_size):
            x = i + batch_size
            print('batch %d:' % (i//batch_size), end=' ', flush=True)
            start_time = time.time()
            batch_color = images[i:x]/255
            batch_grey = utils.to_grayscale(batch_color)
            loss = gen.train_on_batch(batch_grey, batch_color)
            print('loss={:.4f}, time:{:.2f}'.format(loss, time.time()-start_time))
            if ((x <= 10000 and x % 2500 == 0) or x % 5000 == 0):
                gen.save('./regression models/gen%d-%d.h5' % (j, x))

if __name__ == "__main__":               
    #train_gen('./data/unlabeled_X.bin', to_load='0-40000')
    train('./data/unlabeled_X.bin')
