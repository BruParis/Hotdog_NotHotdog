from __future__ import absolute_import, division, print_function, unicode_literals

from image_transformer import ImageTransformer

import os
import random
import pathlib
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
tf.enable_eager_execution()
tf.__version__


img_T = ImageTransformer()

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 64
IMG_HEIGHT = 64
RATIO_TEST = 0.1


def _load_and_process(path):
    img_raw = Image.open(path).convert('RGB')
    img_resized = img_raw.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
    return img_resized


def _export_TFRecord_ds(img_label_ds, tfrecord_file_name):
    with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
        for img, label in img_label_ds:
            img_bytes = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
            }))
            writer.write(example.SerializeToString())


def _format_array(img):
    img_array = np.array(img)[:, :, 0:3]  # Select RGB Channel only
    return img_array


def _fn_ds_item(fn, item):
    return [fn(item[0]), item[1]]


def _pool_map_ds(obj_list, fn):
    pbar = tqdm(total=len(obj_list))
    res = []

    def collect_result(result):
        pbar.update()
        res.append(result)

    pool = Pool()
    for i in range(pbar.total):
        pool.apply_async(_fn_ds_item, args=(
            fn, obj_list[i]), callback=collect_result)

    pool.close()
    pool.join()
    return res


def _load_images(all_img_paths, need_augment=False):
        # load, process, augment and format dataset of image for tensorflow
    print("   -> load")
    img_ds = _pool_map_ds(all_img_paths, _load_and_process)
    if need_augment:
        print("   -> dataset augmentation")
        img_ds = _pool_map_ds(img_ds, img_T.transform_img)
    print("   -> format")
    img_all_ds = _pool_map_ds(img_ds, _format_array)
    return img_all_ds


data_root = pathlib.Path('img')
all_img_paths = [str(path) for path in list(data_root.glob('*/*'))]
img_count = len(all_img_paths)

print('number of img: ' + str(img_count))

# List labels and assign index to each label
LABEL_NAMES = sorted(
    item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(LABEL_NAMES))
print('labels: ' + str(label_to_index))
all_img_labels = [label_to_index[pathlib.Path(path).parent.name]
                  for path in all_img_paths]

item_paths = list(zip(all_img_paths, all_img_labels))
random.shuffle(item_paths)

# Split dataset between test and train
split_idx = int(RATIO_TEST * len(item_paths))
test_items_paths = item_paths[:split_idx]
train_items_paths = item_paths[split_idx:]

print("\nTEST IMAGES: ")
test_img_ds = _load_images(test_items_paths)
print("\nTRAIN IMAGES: ")
train_img_ds = _load_images(train_items_paths, True)

# EXPORT ds
_export_TFRecord_ds(test_img_ds, "test_images.tfrecord")
_export_TFRecord_ds(train_img_ds, "train_images.tfrecord")