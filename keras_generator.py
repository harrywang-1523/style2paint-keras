import numpy as np
from keras_vgg19 import VGG19
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import losses
from keras import backend
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.merge import add

def train():
    sketch = Input((512, 512, 1))
    style = Input((224, 224, 3))
    style512 = Input((512, 512, 3))
    style_grey = Input((512, 512, 1))
    style256 = Input((256, 256, 3))

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

    front_output = front_decoder().fit((conv5, style_grey_img), batch_size=1)
    front_loss = abs(front_output-style_grey)

    conv6 = Conv2D(filters=2048, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv5)
    print("conv6 shape:", conv6.shape)

    vgg_layer = VGG19(input_tensor=style, input_shape=(224, 224, 3)).output
    print('Vgg layer shape:', vgg_layer.shape)

    deconv7 = add([vgg_layer,conv6])
    deconv7 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    print('deconv7(before) shape:', deconv7.shape)
    end_output = end_decoder().fit((deconv7, style512), batch_size=1)
    end_loss = abs(end_output-style512)

    deconv7 = concatenate([deconv7, conv5], axis=3)
    print("deconv7 shape:", deconv7.shape)

    deconv8 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    deconv8 = concatenate([deconv8, conv4], axis=3)
    print('deconv8 shpae:', deconv8.shape)

    deconv9 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv8)
    # crop2 = Cropping2D(cropping=((32,32),(32,32)))(conv2)
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

    model = Model(inputs=[sketch, style, style512, style_grey], outputs=output_layer)

    def expert_loss(y_true, y_pred):
        return abs(y_true - y_pred) + 0.3 * front_loss + 0.9 * end_loss

    model.compile(optimizer=Adam(lr=1e-4), loss=expert_loss(style256, output_layer), metrics=['accuracy'])
    return


def front_decoder():
    input = Input((16, 16, 256))
    style_grey = Input((512, 512, 1))
    guide1 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation='relu')(input)
    print("guide1 shape:", guide1.shape)
    guide2 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide1)
    print("guide2 shape:", guide2.shape)
    guide3 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide2)
    print("guide3 shape:", guide3.shape)
    guide4 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide3)
    print("guide4 shape:", guide4.shape)
    guide5 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide4)
    print("guide5 shape:", guide5.shape)
    output_layer = Dense(1)(guide5)
    print('front decoder shape:', output_layer.shape)
    front_decode_model = Model(inputs=[input, style_grey], output=output_layer)
    front_decode_model.compile(loss=losses.mean_absolute_error(y_true=style_grey, y_pred=output_layer), optimizer='sgd'
                               , metrics=['accuracy'])


def end_decoder():
    input = Input((16, 16, 512))
    style_color = Input((512, 512, 3))
    guide6 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation="relu")(input)
    print("guide6 shape:", guide6.shape)
    guide7 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide6)
    print("guide7 shape:", guide7.shape)
    guide8 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide7)
    print("guide8 shape:", guide8.shape)
    guide9 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide8)
    print("guide9 shape:", guide9.shape)
    guide10 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide9)
    print("guide10 shape:", guide10.shape)
    output_layer = Dense(3)(guide10)
    print('end decoder shape:', output_layer)

    end_decoder_model = Model(inputs=[input, style_color], output=output_layer)
    end_decoder_model.compile(loss=losses.mean_absolute_error(style_color, output_layer), optimizer='sgd'
                              , metrics=['accuracy'])

style_image = load_img('./Style.jpg')
sketch_image = load_img('./Sketch.jpg')
style_image_512 = load_img('./Style(512).jpg')
style_grey_img = load_img('./style_grey.jpg')
style_image_256 = load_img('./style(256).jpg')
print(train().fit((sketch_image, style_image, style_image_512, style_grey_img, style_image_256), batch_size=1))
