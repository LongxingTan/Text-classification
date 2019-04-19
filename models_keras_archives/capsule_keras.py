#code reference
#https://github.com/XifengGuo/CapsNet-Keras

import keras.backend as K
import tensorflow as tf
from keras import initializers,layers

class Capsule_layer(layers.Layer):
    def __init__(self,num_capsule,dim_capsule,routings=3,kernel_initializer='glorot_uniform',**kwargs):
        super(Capsule_layer,self).__init__(**kwargs)
        self.num_capsule=num_capsule
        self.dim_capsule=dim_capsule
        self.routings=routings
        self.kernel_initializer=initializers.get(kernel_initializer)

    def build(self,input_shape):
        assert len(input_shape)>=3
        self.input_num_capsule=input_shape[1]
        self.input_dim_capsule=input_shape[2]

        self.W=self.add_weight(shape=[self.num_capsule,self.input_num_capsule,self.dim_capsule,self.input_dim_capsule],
                               initializer=self.kernel_initializer,name='W')
        self.built=True

    def call(self,inputs,training=None):
        inputs_expand=K.expand_dims(inputs,1)
        inputs_tiled=K.tile(inputs_expand,[1,self.num_capsule,1,1])
        inputs_hat=K.map_fn(lambda x:K.batch_dot(x,self.W,[2,3]),elems=inputs_tiled)

        b=tf.zeros(shape=[K.shape(inputs_hat)[0],self.num_capsule, self.input_num_capsule])
        assert self.routings>0
        for i in range(self.routings):
            c=tf.nn.softmax(b, dim=1)
            outputs=squash(K.batch_dot(c,inputs_hat,[2,2]))
            if i<self.routings-1:
                b+=K.batch_dot(outputs,inputs_hat,[2,3])
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([None,self.num_capsule,self.dim_capsule])

    def get_config(self):
        config={'num_capsule':self.num_capsule,
                'dim_capsule': self.dim_capsule,
                'routings':self.routings}
        base_config=super(Capsule_layer,self).get_config()
        return dict(list(base_config.items())+list(config.items()))




def primary_cap(inputs,dim_capsule,n_channels,kernel_size,strides,padding):
    output=layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size,strides=strides,padding=padding,name='primary_cap')(inputs)
    outputs=layers.Reshape(target_shape=[-1,dim_capsule],name='primary_cap_reshape')(output)
    return layers.Lambda(squash,name='primary_cap_squash')(outputs)

def squash(vectors,axis=-1):
    s_squared_norm=K.sum(K.square(vectors),axis,keepdims=True)
    scale=s_squared_norm/(1+s_squared_norm)/K.sqrt(s_squared_norm+K.epsilon())
    return scale*vectors

def capsnet(input_shape,n_class,routings):
    x=layers.Input(shape=input_shape)
    conv1=layers.Conv2D(filters=256,kernel_size=9, strides=1, padding='valid',activation='relu',name='conv1')(x)
    primarycaper=primary_cap(conv1,dim_capsule=8,n_channels=32,kernel_size=9,strides=2,padding='valid') #[None, num_capsule,dim_capsule]
    digitcaps=Capsule_layer(num_capsule=n_class,dim_capsule=16,routings=routings,name='digitcaps')(primarycaper)
