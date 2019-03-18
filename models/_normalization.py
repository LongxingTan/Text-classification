import tensorflow as tf


class LayerNormalization(tf.layers.Layer):
    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def call(self, x, epsilon=1e-6,name='LN'):
        with tf.variable_scope(name):
            self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],initializer=tf.ones_initializer())
            self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],initializer=tf.zeros_initializer())
            mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
            norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class BatchNormalization(tf.layers.Layer):
    def __init__(self,):
        super(BatchNormalization, self).__init__()

    def call(self, inputs, epsilon=1e-6,name="BN"):
        with tf.variable_scope(name):
            inputs_shape=inputs.get_shape()
            param_shape=inputs_shape[-1:]

            mean,variance=tf.nn.moments(inputs,[-1],keep_dims=True)
            beta=tf.Variable(tf.zeros(param_shape))
            gamma=tf.Variable(tf.ones(param_shape))
            normalized=(inputs-mean)/((variance+epsilon)**(0.5))
            outputs=gamma * normalized + beta
        return outputs


class WeightNormalization(tf.layers.Layer):
    def __init__(self):
        super(WeightNormalization,self).__init__()

    def call(self,inputs,epsilon=1e-6,name="WN"):
        pass