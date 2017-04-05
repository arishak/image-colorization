import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.utils.generic_utils import custom_object_scope
from keras.initializers import Initializer
import keras.losses
# import keras.activations

# constants
IMG_H = 96
BINS = 50
UMAX = 0.43601035
VMAX = 0.61497538

# utility functions
yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])
rgb_from_yuv = np.linalg.inv(yuv_from_rgb)
y_from_rgb_T = np.transpose(np.array([yuv_from_rgb[0]]))
rgb_from_yuv_T = K.cast_to_floatx(np.transpose(rgb_from_yuv))

'''
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
'''

def rgb2yuv(img):
    return np.dot(img, np.transpose(yuv_from_rgb))

def to_grayscale(img):
    return np.dot(img, y_from_rgb_T)

def yuv2rgb(img):
    return np.dot(img, np.transpose(rgb_from_yuv))

def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def load(fname):
    '''
    class yuv2rgb_kernel(Initializer):
        def __init__(self):
            self.rgb_from_yuv = rgb_from_yuv

        def __call__(self, shape, dtype=None):
            return K.reshape(self.rgb_from_yuv, shape)
    '''
    model = None
    with custom_object_scope({}):
        keras.losses.wasserstein = wasserstein
        # keras.activations.___ = ___
        model = load_model(fname + '.h5')
    return model
