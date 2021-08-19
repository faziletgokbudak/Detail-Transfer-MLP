import functools
from tensorflow import keras
from tensorflow.keras import layers, initializers

Dense = functools.partial(layers.Dense, kernel_initializer='uniform',
                          bias_initializer=initializers.constant(0.), activation=layers.LeakyReLU())


class MLP(keras.Model):

    def __init__(self, patch_size):
        super(MLP, self).__init__()
        self.dense1 = Dense(10, name='filters_1')
        self.dense2 = Dense(10, name='filters_2')
        self.dense3 = Dense(10, name='filters_3')
        self.last_layer = Dense(patch_size, name='filters_last')

    def call(self, inputs):
        y = self.dense1(inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        y = self.last_layer(y)

        return y
