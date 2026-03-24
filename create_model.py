import tensorflow as tf
import os
import random
import numpy as np

def create_model(topology = (128, 1), hidden_activation='sigmoid', output_activation='softmax', loss = 'sparse_categorical_crossentropy'):
    # Seed para reprodutibilidade
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    arr_layers = []

    for i, units in enumerate(topology):
        if i != len(topology) - 1:
            arr_layers.append(tf.keras.layers.Dense(units, activation=hidden_activation))
        else:
            arr_layers.append(tf.keras.layers.Dense(units, activation=output_activation))

    model = tf.keras.Sequential(arr_layers)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model