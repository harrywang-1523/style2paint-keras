import tensorflow as tf
import vgg_model
import matplotlib.pyplot as plt
from PIL import Image

# The generator network takes in a sketch and mixes it with the style learned from VGG
# It outputs a picture of size 256 * 256 * 3
def generator(sketch, style):
    vgg = vgg_model.Vgg19(vgg19_npy_path='/Users/Harry/PycharmProjects/style2paint/vgg19.npy')
    return vgg.build(style)

def front_part(sketch):
    K = 16  # first convolutional layer output depth
    L = 32  # second convolutional layer output depth
    M = 64  # third convolutional layer
    N = 128  # fourth convolutional layer
    O = 256  # fifth convolutional layer

    # numbers stand for the size of patch, input channels and output channels
    W1 = tf.Variable(tf.truncated_normal([2, 2, 1, K], stddev=0.1))
    B1 = tf.Variable(tf.ones([K]) / 10.0)
    W2 = tf.Variable(tf.truncated_normal([2, 2, K, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L]) / 10.0)
    W3 = tf.Variable(tf.truncated_normal([2, 2, L, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M]) / 10.0)
    W4 = tf.Variable(tf.truncated_normal([2, 2, M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N]) / 10.0)
    W5 = tf.Variable(tf.truncated_normal([2, 2, N, O], stddev=0.1))
    B5 = tf.Variable(tf.ones([O]) / 10.0)

    Y1 = tf.nn.relu(tf.nn.conv2d(sketch, W1, strides=[1, 2, 2, 1], padding='SAME') + B1)  # 512*512*1 -> 256*256*16
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2)  # 256*256*16 -> 128*128*32
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)  # 128*128*32 -> 64*64*64
    Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, 2, 2, 1], padding='SAME') + B4)  # 64*64*64 -> 32*32*128
    Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, 2, 2, 1], padding='SAME') + B5)  # 32*32*128 -> 16*16*256

    return Y5

def middle_part(front):
    W6 = tf.Variable(tf.truncated_normal([2, 2, 256, 2048], stddev=0.1))
    B6 = tf.Variable(tf.ones([2048]) / 10.0)
    sketch_output = tf.nn.relu(tf.nn.conv2d(front, W6, strides=[1, 2, 2, 1], padding='SAME') + B6)  # 16*16*256 -> 8*8*2048
    return sketch_output

def front_decoder(front):
    W7 = tf.Variable(tf.truncated_normal([2, 2, 256, 512], stddev=0.1))
    B7 = tf.Variable(tf.ones([512]) / 10.0)
    decoder_start = tf.nn.relu(tf.nn.conv2d(front, W7, strides=[1, 2, 2, 1], padding='SAME') + B7)

    D1 = tf.Variable(tf.truncated_normal([2, 2, 256, 512]))
    D2 = tf.Variable(tf.truncated_normal([2, 2, 128, 256]))
    D3 = tf.Variable(tf.truncated_normal([2, 2, 64, 128]))
    D4 = tf.Variable(tf.truncated_normal([2, 2, 32, 64]))
    D5 = tf.Variable(tf.truncated_normal([2, 2, 1, 32]))

    S1 = tf.constant([1, 16, 16, 256])
    S2 = tf.constant([1, 32, 32, 128])
    S3 = tf.constant([1, 64, 64, 64])
    S4 = tf.constant([1, 128, 128, 32])
    S5 = tf.constant([1, 256, 256, 1])

    Z1 = tf.nn.conv2d_transpose(decoder_start, D1, strides=[1, 2, 2, 1], output_shape=S1, padding='SAME')
    Z2 = tf.nn.conv2d_transpose(Z1, D2, strides=[1, 2, 2, 1], output_shape=S2, padding='SAME')
    Z3 = tf.nn.conv2d_transpose(Z2, D3, strides=[1, 2, 2, 1], output_shape=S3, padding='SAME')
    Z4 = tf.nn.conv2d_transpose(Z3, D4, strides=[1, 2, 2, 1], output_shape=S4, padding='SAME')
    Z5 = tf.nn.conv2d_transpose(Z4, D5, strides=[1, 2, 2, 1], output_shape=S5, padding='SAME')
    return tf.reshape(Z5, [256,256,1])
