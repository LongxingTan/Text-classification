import tensorflow as tf
from models._embedding import Embedding_layer
from models._model_params import *


class Capsule(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])
        self.capsule_layer_conv=Capsule_layer()
        self.capsule_layer_fc=Capsule_layer()

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.train:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)

        embedding_outputs=tf.expand_dims(embedding_outputs,-1)
        with tf.name_scope('conv'):
            conv1 = tf.layers.conv2d(inputs=embedding_outputs,
                                     filters=params['filters'],
                                     kernel_size=[kernel_size],
                                     strides=1,
                                     padding='valid',
                                     activation=tf.nn.relu)

        capsule_output=self.capsule_layer_conv(embedding_outputs)
        capsule_output_flat=self.capsule_flatten(capsule_output)
        self.logits=self.capsule_layer_fc(capsule_output_flat)

    def capsule_flatten(self):
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits


class Capsule_layer(tf.layers.Layer):
    def __init__(self,hidden_size,if_routing):
        super(Capsule_layer,self).__init__()
        self.hidden_size=hidden_size
        self.if_routing=if_routing

    def squash(self,vector,axis=-1,epsilon=1e-7):
        vec_squared_norm=tf.reduce_sum(tf.square(vector),axis=axis,keep_dims=True)
        scale=vec_squared_norm/(1+vec_squared_norm)/tf.sqrt(vec_squared_norm+epsilon)
        return scale*vector

    def dynamic_routing(self,input,iters):
        b=tf.zeros_like(input[:,:,:,0])
        for i in range(iters):
            c=tf.nn.softmax(b,1)
            output=self.squash(tf.tensordot(c,input,axes=[2,2]))
            if i<iters-1:
                b=b+tf.tensordot(output,input,axes=[2,3])
        return output


class Capsule_layer_conv(Capsule_layer):
    def __init__(self):
        super(Capsule_layer_conv,self).__init__()

    def call(self, inputs, **kwargs):
        pass

class Capsule_layer_fc(Capsule_layer):
    def __init__(self):
        super(Capsule_layer_fc,self).__init__()

    def call(self, inputs, **kwargs):
        pass