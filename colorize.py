import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from stl10_input import read_all_images
import numpy as np
import keras.initializers
from keras.layers import Input
from keras.models import load_model, Model
import keras.backend as K
from keras.utils.generic_utils import custom_object_scope
import keras.losses, keras.activations
from keras.layers.advanced_activations import LeakyReLU
from skimage import color
from matplotlib import cm

fname = './models/gen0-2500.h5'
images = read_all_images('./data/stl10_binary/test_X.bin')
colorizer = utils.load(fname)
disc = utils.load(fname.replace('gen', 'disc'))

def gen_images(n):
    #np.random.seed(0)
    idxs = np.random.randint(0, len(images), size=n)
    real = images[idxs]/255
    grey = utils.to_grayscale(real)
    gen = colorizer.predict(grey)
    grey = np.repeat(grey, 3, axis=3)
    #print(colorizer.evaluate(grey/255, real))
    print(np.min(gen), np.max(gen))
    return (grey, gen, real)

def plot_images(rows=4, cols=4):
    imgs = gen_images(rows*cols)
    fig = plt.figure(figsize=(18, 7))
    outer = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.2)

    for i in range(rows*cols):
        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[i],
                                                 wspace=0.05, hspace=0.05)
        for j in range(3):
            ax = plt.Subplot(fig, inner[j])
            ax.axis('off')
            img = imgs[j][i]
            if len(img.shape) == 2:
                ax.imshow(np.clip(img, 0, 255), cmap='gray')
            else:
                ax.imshow(img)
            fig.add_subplot(ax)

    outer.tight_layout(fig)
    plt.show()

def test_disc(n=20):
    (grey, gen, real) = gen_images(n)
    print(disc.predict(grey))
    print(disc.predict(gen))
    print(disc.predict(real))

def test_gan(n=20, img_h=96):
    (grey, gen, real) = gen_images(n)
    ins = Input(shape=(img_h, img_h, 1))
    generated = colorizer(ins)
    outs = disc(generated)
    GAN = Model(inputs=ins, outputs=outs)
    print(GAN.predict(np.reshape(grey, (n, 96, 96, 1))/255))
    
plot_images()
test_disc()
#test_gan()

def test_convert():
    img = mpimg.imread('test.jpg')
    cmap_grey = cm.get_cmap('gray')
    grey = utils.to_grayscale(img)/255
    grey = np.repeat(grey, 3, axis=2)
    print(grey)
    plt.imshow(img)
    plt.show()

#test_convert()
