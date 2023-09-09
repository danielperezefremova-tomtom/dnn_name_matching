import tensorflow as tf
from keras.activations import relu

def get_convolutions_pool(x_input, max_len, suffix, n_grams=[2,3,5,8, 13], feature_maps=128):

    branches = []
    for n in n_grams:
        branch = tf.keras.layers.Conv1D(filters=feature_maps, kernel_size=n, activation=relu,
                        name='Conv_' + suffix + '_' + str(n))(x_input)
        branch = tf.keras.layers.MaxPooling1D(pool_size=max_len - n + 1,
                                              strides=1, padding='valid',
                              name='MaxPooling_' + suffix + '_' + str(n))(branch)
        branch = tf.keras.layers.Flatten(name='Flatten_' + suffix + '_' + str(n))(branch)
        branches.append(branch)
    return branches
