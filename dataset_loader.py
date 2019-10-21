import os
import random
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()
tf.__version__


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_WIDTH = 64
IMG_HEIGHT = 64

# feature description corresponding to build_dataset export function
IMG_FEATURES_DESC = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _prepare_dataset(ds, cache=True, shuffle_buffer_size=1000):
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def _show_img(img_raw):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_raw)
    plt.axis('off')
    plt.show(block=True)


def _parse(ds_item):
    return tf.io.parse_single_example(ds_item, IMG_FEATURES_DESC)


def _parse_decode(ds_item):
    parsed_example = tf.parse_single_example(ds_item, IMG_FEATURES_DESC)
    label = tf.cast(parsed_example['label'], tf.float32)

    img_shape = tf.stack([IMG_WIDTH, IMG_HEIGHT, 3])
    img = tf.decode_raw(parsed_example['image_raw'], tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, img_shape)
    # set to range [-1, 1] for keras
    img = tf.subtract(tf.multiply(img, 2 / 255.), 1)
    return img, label


def _show_img_ds(img_parsed_ds, start_index=None):
    end_index = start_index + 5
    for idx, item in tf.data.Dataset.enumerate(img_parsed_ds):
        tf.print("idx: ", idx)
        if idx < start_index:
            continue

        img = item['image_raw'].numpy()
        img_array = np.frombuffer(img, dtype=np.uint8)
        img_array.shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        print(str(img_array))
        _show_img(img_array)

        if idx > end_index:
            break


def load_dataset(filename):
    # LOAD dataset from .tfrecord files
    img_ds = tf.data.TFRecordDataset(filename)

    # img_parsed_ds = img_ds.map(_parse)
    # _show_img_ds(img_parsed_ds, 0)
    count = 0
    for _ in img_ds:
      count += 1
    
    img_decoded_ds = img_ds.map(_parse_decode)

    # Prepare dataset - BATCH + Setting shuffle buffer
    train_ds = _prepare_dataset(img_decoded_ds)

    return train_ds, count
