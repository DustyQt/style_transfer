import os
import io
import tensorflow as tf
from flask import send_file
from models.style_content_model import StyleContentModel
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
from utils import load_img, tensor_to_image, clip_0_1
class StyleTransferProcesor:
    def __init__(self, logger, upload_folder):
        self.logger = logger
        self.upload_folder = upload_folder

    def call_model(self, content_path, style_path, hyper_parms=None):
        self.logger.debug(f' type of content : {type(content_path)}')
        content_image = load_img(f'{self.upload_folder}/{content_path}')
        style_image = load_img(f'{self.upload_folder}/{style_path}')
        extractor = StyleContentModel()

        extractor.set_targets(style_image, content_image)
        #initialize image with content image
        image = tf.Variable(content_image)
        #optimizer 
        opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        #weight of each loss
        style_weight=1e-2
        content_weight=1e4
        extractor.set_loss_weights(style_weight, content_weight)

        start = time.time()
        epochs = 10
        steps_per_epoch = 100

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = extractor.style_content_loss(outputs)

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
                print(".", end='', flush=True)
        print("Train step: {}".format(step))
        end = time.time()
        print("Total time: {:.1f}".format(end-start))

        img_bytes = io.BytesIO()
        plt.imshow(tensor_to_image(image))
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)

        return send_file(
            img_bytes,
            mimetype='image/png',
            as_attachment=True,
            download_name='result.jpg',
        )
