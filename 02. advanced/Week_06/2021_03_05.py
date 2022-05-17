#%% Custom Layer (SPP)

import tensorflow as tf

class SpatialPyramidPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_list, *args, **kwargs):
        super().__init__(pool_list, *args, **kwargs)
        self.pool_list = pool_list
    
    def build(self, input_shape):
        pass
    
    def call(self, input_):
        for i in self.pool_list:
            print(input_.shape)
            print(tf.reduce_max(input_))
        return tf.reduce_max(input_)
    
#%%


