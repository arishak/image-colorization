import numpy as np
from keras.layers import Input
from keras.layers.core import Dense, Reshape, Lambda
from keras.layers.convolutional import Conv1D, Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.generic_utils import custom_object_scope
from keras.initializers import Initializer
import keras.losses
import keras.activations
from keras.optimizers import Adam
import keras.backend as K
import time
from stl10_input import read_all_images
#import skimage.io

img_h = 96
bins = 50
umax = 0.43601035
vmax = 0.61497538

yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])
y_from_rgb = np.transpose(np.array([yuv_from_rgb[0]]))
rgb_from_yuv = K.cast_to_floatx(np.transpose(np.linalg.inv(yuv_from_rgb)))

#img = skimage.io.imread('test.jpg')
    
def bin2u(x):
    x = K.cast(x, K.floatx())
    2*umax*(x+0.5)/bins - umax
    return x

def bin2v(x):
    x = K.cast(x, K.floatx())
    2*vmax*(x+0.5)/bins - vmax
    return x

def tanh_u(x):
    return umax*K.tanh(x)

def tanh_v(x):
    return vmax*K.tanh(x)

def tall_sigmoid(x, scale=255):
    return scale*K.sigmoid(x)

def wide_sigmoid(x, scale=2):
    return K.sigmoid(x/scale)

def wide_tanh(x, scale=2):
    return K.sigmoid(x/scale)

def yuv2rgb_kernel(shape, dtype=None):
    print(shape)
    return K.reshape(rgb_from_yuv, shape)

def rgb2yuv(img):
    return np.dot(img, np.transpose(yuv_from_rgb))
    
yuv2rgbLayer = Conv2D(3, (1, 1), trainable=False, use_bias=False,
                      kernel_initializer=yuv2rgb_kernel)

def get_test(img_h=img_h):
    inputs = Input(shape=(img_h, img_h, 3))
    x = yuv2rgbLayer(inputs)
    return Model(inputs=inputs, outputs=x)

def get_gen(img_h):
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
    '''
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    out1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    out1 = Conv2D(8, (3, 3), activation='relu', padding='same')(out1)
    out1 = Conv2D(1, (3, 3), activation='relu', padding='same')(out1)
    out2 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    out2 = Conv2D(8, (3, 3), activation='relu', padding='same')(out2)
    out2 = Conv2D(1, (3, 3), activation='relu', padding='same')(out2)
    
    x = Reshape((img_h*img_h, 64))(x)
    out1 = Conv1D(bins, (1,), activation='softmax', padding='same')(x)
    out1 = Lambda(K.argmax)(out1)
    out1 = Lambda(bin2u)(out1)
    out1 = Reshape((img_h, img_h, 1))(out1)
    out2 = Conv1D(bins, (1,), activation='softmax', padding='same')(x)
    out2 = Lambda(K.argmax)(out2)
    out2 = Lambda(bin2v)(out2)
    out2 = Reshape((img_h, img_h, 1))(out2)
    '''
    x = Conv2D(2, (3, 3), activation='elu', padding='same')(x)
    out = Concatenate()([inputs, x])
    out = Conv2D(3, (1, 1), activation='tanh', padding='same')(out)
    #out_u = Conv2D(1, (3, 3), activation=tanh_u, padding="same")(x)
    #out_v = Conv2D(1, (3, 3), activation=tanh_v, padding="same")(x)
    #out = yuv2rgbLayer(out)
    out = Conv2D(3, (1, 1), activation="sigmoid")(out)
    return Model(inputs=inputs, outputs=out)

def get_disc(img_h):
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
    ins = Input(shape=(img_h, img_h, 1))
    generated = gen(ins)
    outs = disc(generated)
    return Model(inputs=ins, outputs=outs)
    
def model_info(model, filename='model.png'):
    print(model.summary())
    plot_model(model, to_file=filename, show_shapes=True)

def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def load(fname):
    class yuv2rgb_kernel(keras.initializers.Initializer):
        def __init__(self):
            self.rgb_from_yuv = rgb_from_yuv

        def __call__(self, shape, dtype=None):
            return K.reshape(self.rgb_from_yuv, shape)
    
    with custom_object_scope({"yuv2rgb_kernel":yuv2rgb_kernel}):
        keras.losses.wasserstein = wasserstein
        keras.activations.wide_tanh = wide_tanh
        keras.activations.wide_sigmoid = wide_sigmoid
        keras.activations.tall_sigmoid = tall_sigmoid
        model = load_model(fname + '.h5')
    return model
    
def train(fname, epochs=1, batch_size=50, to_load=None):
    images = read_all_images(fname)

    # get models and compile
    gen = get_gen(img_h)
    gen.compile(loss='mean_absolute_error', optimizer='nadam')

    disc = get_disc(img_h)
    disc.trainable = True
    disc.compile(loss=wasserstein, optimizer='nadam')
    disc._make_train_function()
    
    if to_load:
        if 'gen' in to_load:
            gen = load(to_load['gen'])
            print("loaded generator")
        if 'disc' in to_load:
            disc = load(to_load['disc'])
            print("loaded discriminator")
    
    disc.trainable = False
    GAN = get_gan(gen, disc)
    print(GAN.summary())
    GAN.compile(loss=wasserstein, optimizer='nadam')
    ones = np.ones(batch_size)
    mones = -ones
    mixones = np.concatenate((mones, ones))

    # test save / load
    GAN.save('./models/GAN-test.h5')
    load('./models/GAN-test')
    
    for j in range(epochs):
        np.random.shuffle(images)
        for i in range(0, len(images), batch_size):
            x = i + batch_size
            print('batch %d:' % (i//batch_size), end=' ', flush=True)
            start_time = time.time()
            batch_color = images[i:x]/255
            batch_grey = np.dot(batch_color, y_from_rgb)
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
        gen = load('./regression models/gen' + to_load)
    else:
        gen = get_gen(img_h)
        gen.compile(loss='mean_squared_error', optimizer="nadam")
    images = read_all_images(fname)
    
    for j in range(epochs):
        np.random.shuffle(images)
        for i in range(0, len(images), batch_size):
            x = i + batch_size
            print('batch %d:' % (i//batch_size), end=' ', flush=True)
            start_time = time.time()
            batch_color = images[i:x]/255
            #batch_yuv = rgb2yuv(batch_rgb)
            batch_grey = np.dot(batch_color, y_from_rgb)
            loss = gen.train_on_batch(batch_grey, batch_color)
            print('loss={:.4f}, time:{:.2f}'.format(loss, time.time()-start_time))
            if ((x <= 10000 and x % 2500 == 0) or x % 5000 == 0):
                gen.save('./regression models/gen%d-%d.h5' % (j, x))
            
#train_gen('./data/unlabeled_X.bin', to_load='0-40000')
train('./data/unlabeled_X.bin', to_load=None)
