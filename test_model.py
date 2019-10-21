from __future__ import absolute_import, division, print_function, unicode_literals

import os
from dataset_loader import *
from model import *
import tensorflow as tf
tf.enable_eager_execution()
tf.__version__

checkpoint_path = 'training_w.cpkt'
test_ds, test_c = load_dataset('test_images.tfrecord')
model = create_model()

test_steps = 0.1 * int(test_c / BATCH_SIZE)
loss, acc = model.evaluate(test_ds, steps=test_steps)
print('Untrained: ' + str(acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_ds, steps=test_steps)
print('-> Trained: ' + str(acc))