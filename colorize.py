import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from stl10_input import read_all_images
import utils
import models

fname = './models/saves/gen0-30000'
images = read_all_images('./data/test_X.bin')
colorizer = utils.load(fname)
disc = utils.load(fname.replace('gen', 'disc'))
#disc = utils.load('./models/saves/disc0-100000')
print('loaded')

def gen_images(n):
    #np.random.seed(0)
    idxs = np.random.randint(0, len(images), size=n)
    real = images[idxs]/255
    grey = utils.to_grayscale(real)
    gen = colorizer.predict(grey)
    print(gen)
    grey = np.repeat(grey, 3, axis=3)
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
            ax.imshow(imgs[j][i])
            fig.add_subplot(ax)

    outer.tight_layout(fig)
    plt.show()

def test_disc(n=20):
    (grey, gen, real) = gen_images(n)
    print(disc.predict(grey))
    print(disc.predict(gen))
    print(disc.predict(real))

def test_gan(n=20, img_h=utils.IMG_H):
    (grey, gen, real) = gen_images(n)
    GAN = models.get_gan(colorizer, disc)
    print('GAN', GAN.predict(grey[...,0:1]))

def test_convert():
    img = mpimg.imread('test.jpg')
    cmap_grey = cm.get_cmap('gray')
    grey = utils.to_grayscale(img)/255
    grey = np.repeat(grey, 3, axis=2)
    print(grey)
    plt.imshow(img)
    plt.show()

plot_images()
test_disc()
#test_gan()
