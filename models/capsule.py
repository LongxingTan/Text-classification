import tensorflow as tf
from models._embedding import Embedding_layer
from model_params import params


class Capsule(object):
    def __init__(self, training):
        self.training = training
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'],
                                               embedding_type=params['embedding_type'])
        self.capsule_layer_conv=Capsule_layer_conv(shape=[3,1,128,16],vec_length=16)
        self.capsule_layer_dense=Capsule_layer_dense(hidden_size=params['n_class'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)
        if self.training:
            embedding_outputs = tf.nn.dropout(embedding_outputs, params['embedding_dropout_keep'])

        embedding_outputs=tf.expand_dims(embedding_outputs,-1)
        with tf.name_scope('conv'):
            conv1 = tf.layers.conv2d(inputs=embedding_outputs,
                                     filters=params['filters'],
                                     kernel_size=[3,300],
                                     strides=1,
                                     padding='valid',
                                     activation=tf.nn.relu) #[batch_size,seq_length-2,1,128]

        cap_conv=self.capsule_layer_conv(conv1)
        cap_conv_shape=cap_conv.get_shape().as_list()
        cap_flat=tf.reshape(cap_conv,[cap_conv_shape[0],-1,self.capsule_layer_conv.vec_length])
        print(cap_flat.get_shape().as_list())
        self.logits=self.capsule_layer_dense(cap_flat)


    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits


class Capsule_layer(tf.layers.Layer):
    def __init__(self):
        super(Capsule_layer,self).__init__()

    def squash(self,vector,axis=-1,epsilon=1e-7):
        vec_squared_norm=tf.reduce_sum(tf.square(vector),axis=axis,keepdims=True)
        scale=vec_squared_norm/(1+vec_squared_norm)/tf.sqrt(vec_squared_norm+epsilon)
        return scale*vector

    def dynamic_routing(self,input,iters):
        print('input shape', input.get_shape().as_list())
        print('b shape', input.get_shape().as_list()[:-1])
        b=tf.zeros(shape=input.get_shape().as_list()[:-1], dtype=tf.float32, name='b')
        for i in range(iters):
            c=tf.nn.softmax(b,1)
            output=self.squash(tf.tensordot(c,input,axes=[2,2]))
            if i<iters-1:
                b=b+tf.tensordot(output,input,axes=[2,3])
        return output

    def vec_transform_conv(self,inputs,input_cap_dim,input_cap_num,output_cap_dim,output_cap_num):
        kernel_size=[1]
        u_hat_vecs=tf.layers.conv1d(inputs=inputs,kernel_size=kernel_size,filters=output_cap_dim*output_cap_num)
        u_hat_vecs=tf.reshape(u_hat_vecs,[-1,input_cap_num,output_cap_num,output_cap_dim])
        u_hat_vecs=tf.transpose(u_hat_vecs,(0,2,1,3))
        print('u_hat shape',u_hat_vecs.get_shape().as_list())
        return u_hat_vecs


class Capsule_layer_conv(Capsule_layer):
    def __init__(self,shape,vec_length):
        super(Capsule_layer_conv,self).__init__()
        self.shape=shape
        self.vec_length=vec_length

    def call(self, inputs, **kwargs):
        cap=tf.layers.conv2d(inputs,filters=self.shape[-1]*self.vec_length,kernel_size=self.shape[:2],strides=1)
        print('cap1 shape',cap.get_shape().as_list()) #[batch_size,seq_length-4,1,256]
        cap_shape=cap.get_shape().as_list()
        cap=tf.reshape(cap,cap_shape[0:2]+[-1,self.vec_length])
        cap=self.squash(cap)
        print('cap2 shape', cap.get_shape().as_list())
        return cap

class Capsule_layer_dense(Capsule_layer):
    def __init__(self,hidden_size):
        super(Capsule_layer_dense,self).__init__()
        self.hidden_size=hidden_size

    def call(self, inputs, **kwargs):
        inputs_shape=inputs.get_shape().as_list()
        cap=self.vec_transform_conv(inputs,input_cap_dim=inputs_shape[-1],input_cap_num=inputs_shape[1],
                                    output_cap_dim=inputs_shape[-1],output_cap_num=self.hidden_size)
        output=self.dynamic_routing(cap,iters=3)
        print('out shape',output.get_shape().as_list())
        output=tf.sqrt(tf.reduce_sum(tf.square(output),axis=2))
        print('output shape2', output.get_shape().as_list())
        return output