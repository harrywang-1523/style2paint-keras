import tensorflow as tf
import os
import cv2
import numpy as np
import Generator
from scipy import misc
import matplotlib.pyplot as plt

VGG = "/Users/Harry/PycharmProjects/style2paint/vgg19.npy"
style_image_path = "/Users/Harry/PycharmProjects/style2paint/Style.jpg"
sketch_image_path = '/Users/Harry/PycharmProjects/style2paint/Sketch.jpg'
if os.path.exists(VGG):
    print('VGG-19 Model is ready to use')
else:
    print("Failed to load VGG-19.mat")

if os.path.exists(sketch_image_path):
    print('Sketch image has been loaded successfully')
else:
    print('Failed to load sketch image')
if os.path.exists(style_image_path):
    # style_image = tf.constant(cv2.imread(style_image_path), dtype=tf.float32)
    # style_image = tf.reshape(style_image, shape=[-1, 512, 512, 3])
    print('Style image has been loaded successfully')
else:
    print('Failed to load style image')

sketch_image = tf.placeholder(tf.float32, [1, 512, 512, 1])
style_image = tf.placeholder(tf.float32, [1, 224, 224, 3])
sketch_output = Generator.generator(sketch_image, style_image)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sketch = cv2.imread(sketch_image_path, cv2.IMREAD_GRAYSCALE).reshape([-1, 512, 512, 1])
    style = cv2.imread(style_image_path).reshape([-1, 224, 224, 3])
    out = sess.run(sketch_output, feed_dict={sketch_image: sketch, style_image: style})
    # out = sess.run(sketch_output, feed_dict={sketch_image:sketch, style_image:style})
