import numpy as np
import keras.backend as K
import keras.losses

class Constants(object):
    def __init__(self):
        IMG_H = 96
        BINS = 50
        UMAX = 0.43601035
        VMAX = 0.61497538

yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])
rgb_from_yuv = np.linalg.inv(yuv_from_rgb)
y_from_rgb_T = np.transpose(np.array([yuv_from_rgb[0]]))
rgb_from_yuv_T = K.cast_to_floatx(np.transpose(rgb_from_yuv))

class Functions(object):
    def rgb2yuv(img):
        return np.dot(img, np.transpose(yuv_from_rgb))

    def yuv2rgb(img):
        return np.dot(img, np.transpose(rgb_from_yuv))

    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def load(fname):
        with custom_object_scope({}):
            keras.losses.wasserstein = wasserstein
            model = load_model(fname + '.h5')
