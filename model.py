import tensorflow as tf

def create_model():
    mobile_net = tf.keras.applications.MobileNetV2(
        input_shape=(64, 64, 3), include_top=False)
    mobile_net.trainable = True
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2)])  # Hardcoded-param

    # Compile model to describe training procedure
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])
    model.summary()
    return model