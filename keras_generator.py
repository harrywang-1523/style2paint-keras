import numpy as np
from keras_vgg19 import VGG19
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import losses
from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.merge import add

def main_model():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))

    # The convolutional layers in the front
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    print("conv1 shape:", conv1.shape)

    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    print("conv2 shape:", conv2.shape)

    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    print("conv3 shape:", conv3.shape)

    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    print("conv4 shape:", conv4.shape)

    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    print("conv5 shape:", conv5.shape)

    conv6 = Conv2D(filters=2048, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv5)
    print("conv6 shape:", conv6.shape)

    # resize to 224
    vgg_style = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(style)
    vgg_layer = VGG19(input_tensor=vgg_style, input_shape=(224, 224, 3)).output
    print('Vgg layer shape:', vgg_layer.shape)

    deconv7 = add([vgg_layer,conv6])
    deconv7 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    print('deconv7(before) shape:', deconv7.shape)

    deconv7 = concatenate([deconv7, conv5], axis=3)
    print("deconv7 shape:", deconv7.shape)

    deconv8 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    deconv8 = concatenate([deconv8, conv4], axis=3)
    print('deconv8 shpae:', deconv8.shape)

    deconv9 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv8)
    deconv9 = concatenate([deconv9, conv3], axis=3)
    print('deconv9 shpae:', deconv9.shape)

    deconv10 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv9)
    deconv10 = concatenate([deconv10, conv2], axis=3)
    print('deconv10 shape:', deconv10.shape)

    deconv11 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv10)
    deconv11 = concatenate([deconv11, conv1], axis=3)
    print('deconv11 shape:', deconv11.shape)

    output_layer = Dense(units=3)(deconv11)
    print('output shape:', output_layer.shape)

    style_256 = Lambda(lambda image: ktf.image.resize_images(image, (256, 256)))(style)
    model = Model(inputs=[sketch, style], outputs=output_layer)
    model.compile(optimizer="sgd", loss=losses.mean_absolute_error(style_256, output_layer), metrics=['accuracy'])


def end_decoder():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    conv6 = Conv2D(filters=2048, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv5)
    vgg_layer = VGG19(input_tensor=style, input_shape=(224, 224, 3)).output
    deconv7 = add([vgg_layer, conv6])
    deconv7 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    guide6 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation='relu')(deconv7)
    guide7 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide6)
    guide8 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide7)
    guide9 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide8)
    guide10 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide9)
    output_layer = Dense(3)(guide10)

    front_decode_model = Model(inputs=[sketch, style], output=output_layer)
    front_decode_model.compile(loss=losses.mean_absolute_error(style, output_layer), optimizer='sgd', metrics=['accuracy'])


def front_decoder():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    guide1 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation="relu")(conv5)
    guide2 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide1)
    guide3 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide2)
    guide4 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide3)
    guide5 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide4)
    output_layer = Dense(1)(guide5)

    grey_style = ktf.image.rgb_to_grayscale(style)
    end_decoder_model = Model(inputs=[sketch, style], output=output_layer)
    end_decoder_model.compile(loss=losses.mean_absolute_error(grey_style, output_layer), optimizer='sgd', metrics=['accuracy'])


def train():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))
    network_output = main_model().fit((sketch, style), batch_size=1)
    network_loss = main_model().loss
    front_output = front_decoder().fit((sketch, style), batch_size=1)
    front_loss = front_decoder().loss
    end_output = end_decoder().fit((sketch, style), batch_size=1)
    end_loss = end_decoder().loss
    end_decoder_model = Model(inputs=[sketch, style], output=None)
    end_decoder_model.compile(loss=front_loss + end_loss + network_loss, optimizer='sgd', metrics=['accuracy'])

sketch_image = load_img('./Sketch.jpg')
style_image= load_img('./Style(512).jpg')
print(train().fit((sketch_image, style_image), batch_size=1))
