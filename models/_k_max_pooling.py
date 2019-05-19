import tensorflow as tf


class KMaxPooling(tf.layers.Layer):
    #  K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    def __init__(self):
        super(KMaxPooling,self).__init__()

    def call(self, inputs,k=1,name='K_max_pooling'):
        with tf.variable_scope(name):
            shifted_input = tf.transpose(inputs,
                                         [0, 2, 1])  # batch_size * seq_length * filters => batch * filters * seq_length
            top_k = tf.nn.top_k(shifted_input, k=k, sorted=True, name=None)[0]  # => batch*filters * k
            top_k = tf.transpose(top_k, [0, 2, 1])
        return top_k
