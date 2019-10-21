from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import time
import random
import pathlib
from dataset_loader import *
from model import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

tf.enable_eager_execution()
tf.__version__

LABEL_NAMES = ['hotdog', 'random']


def show_batch(image_batch, label_batch):
    print("SHOW BATCH")
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        img = ((image_batch[n].numpy() + 1) * 127.5).astype(np.uint8)
        plt.imshow(img)
        plt.title(LABEL_NAMES[label_batch[n]][0].title())
        plt.axis('off')
    plt.show(block=True)


def _plot_history(hist):
    print('history: ' + str(hist.history))
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 6])
    plt.legend(loc='lower right')
    plt.show(block=True)


train_ds, train_c = load_dataset('train_images.tfrecord')
image_batch, label_batch = next(iter(train_ds))

model = create_model()
logit_batch = model(image_batch).numpy()
print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print("Shape:", logit_batch.shape)

training_steps = int(train_c / BATCH_SIZE)
# Callback to save model's weights -> Keras checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    "training_w.cpkt", save_weights_only=True, verbose=1)
history = model.fit(train_ds, epochs=7, steps_per_epoch=training_steps, callbacks=[cp_callback])
_plot_history(history)