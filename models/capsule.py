import tensorflow as tf
from models._embedding import Embedding_layer
from model_params import params


class Capsule(object):
    def __init__(self, training):
        self.training = training
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'],
                                               embedding_type=params['embedding_type'])
        self.capsule_layer_conv=Capsule_layer_conv(shape=[3,1,128,16],vec_len=16)
        self.capsule_layer_fc=Capsule_layer_dense(hidden_size=params['n_class'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.training:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)

        embedding_outputs=tf.expand_dims(embedding_outputs,-1)
        with tf.name_scope('conv'):
            conv1 = tf.layers.conv2d(inputs=embedding_outputs,
                                     filters=params['filters'],
                                     kernel_size=[3,300],
                                     strides=1,
                                     padding='valid',
                                     activation=tf.nn.relu) #[batch_size,seq_length-2,1,128]
            print(conv1.get_shape().as_list())

        cap_conv=self.capsule_layer_conv(conv1)
        #capsule_output_flat=self.capsule_flatten(cap_conv)
        self.logits=self.capsule_layer_fc(cap_conv)

    def capsule_flatten(self,input):
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits


class Capsule_layer(tf.layers.Layer):
    def __init__(self):
        super(Capsule_layer,self).__init__()


    def squash(self,vector,axis=-1,epsilon=1e-7):
        vec_squared_norm=tf.reduce_sum(tf.square(vector),axis=axis,keep_dims=True)
        scale=vec_squared_norm/(1+vec_squared_norm)/tf.sqrt(vec_squared_norm+epsilon)
        return scale*vector

    def dynamic_routing(self,input,iters):
        b=tf.zeros(shape=input.get_shape().as_list()[:-1], dtype=tf.float32, name='b')
        print('b shape', b.get_shape().as_list())
        for i in range(iters):
            c=tf.nn.softmax(b,1)
            output=self.squash(tf.tensordot(c,input,axes=[2,2]))
            if i<iters-1:
                b=b+tf.tensordot(output,input,axes=[2,3])
        return output


class Capsule_layer_conv(Capsule_layer):
    def __init__(self,shape,vec_len):
        super(Capsule_layer_conv,self).__init__()
        self.shape=shape
        self.vec_len=vec_len

    def call(self, inputs, **kwargs):
        cap=tf.layers.conv2d(inputs,filters=self.shape[-1]*self.vec_len,kernel_size=self.shape[:2],strides=1)
        print('cap1 shape',cap.get_shape().as_list()) #[batch_size,seq_length-4,1,256]
        cap=tf.reshape(cap,[params['batch_size'],-1,self.vec_len,1])
        cap=self.squash(cap)
        print('cap2 shape', cap.get_shape().as_list())
        return cap

class Capsule_layer_dense(Capsule_layer):
    def __init__(self,hidden_size):
        super(Capsule_layer_dense,self).__init__()
        self.hidden_size=hidden_size

    def call(self, inputs, **kwargs):
        #input=tf.reshape(inputs,[])
        cap=self.dynamic_routing(inputs,iters=3)
        return cap