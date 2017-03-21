from keras.layers import Input
from keras.layers.core import Dense, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add, Concatenate
from keras.models import Model
from keras.utils import plot_model

img_h = 256
bins = 50

inputs = Input(shape=(img_h, img_h, 1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
concat1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(concat1)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
sum1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(sum1)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
sum2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(sum2)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
concat4 = Conv2D(256, (1, 1), activation='relu', padding='same')(x)
x = Add()([sum2, UpSampling2D()(concat4)])
concat3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Add()([sum1, UpSampling2D()(concat3)])
concat2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Concatenate()([concat1, UpSampling2D()(concat2), UpSampling2D(4)(concat3), UpSampling2D(8)(concat4)])
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Reshape((img_h*img_h, 64))(x)
print(x.shape)
out1 = Conv1D(bins, (1,), activation='softmax', padding='same')(x)
out1 = Reshape((img_h, img_h, bins))(out1)
out2 = Conv1D(bins, (1,), activation='softmax', padding='same')(x)
out1 = Reshape((img_h, img_h, bins))(out2)

model = Model(inputs=inputs, outputs=[out1, out2])

print(out1.shape)
print(model.summary())

plot_model(model, to_file='model.png', show_shapes=True)
