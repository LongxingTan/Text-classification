
import tensorflow as tf
epsilon = 1e-9

class Capsule_layer:
    def __init__(self,n_vector,if_routing=True,layer_type='FC'):
        self.n_vector=n_vector
        self.if_routing=if_routing
        self.layer_type=layer_type

    def __call__(self,input,output_cap_num=None,kernel_size=None,stride=None, padding=None,iters=3):
        '''[filter_height, filter_width, in_channels, out_channels]'''
        if self.layer_type=='CONV':
            self.kernel_size=kernel_size
            self.stride=stride
            self.padding=padding

            if not self.if_routing:
                filter_size=kernel_size[0:-1]+[self.n_vector*kernel_size[-1]]
                capsules=tf.nn.conv2d(input,filter=filter_size,strides=self.stride,padding=self.padding,name='cap_conv')
                capsules=tf.nn.relu(capsules)
                capsules_shape=capsules.get_shape().as_list()
                capsules=tf.reshape(capsules,[capsules_shape[0],-1,self.n_vector,1]) #flat
                capsules=self.squash(capsules)
                return capsules

        if self.layer_type=='FC':
            if self.if_routing:
                input_shape=input.get_shape().as_list()
                self.input=tf.reshape(input,shape=[input_shape[0],-1,input_shape[-1]])
                u_hat_vector=self.vec_transform(self.input,input_shape[-1],input_shape[1],input_shape[-1],output_cap_num)

                with tf.variable_scope('routing'):
                    capsules=self.dynamic_routing1(u_hat_vector,iters=iters)
                    capsules=tf.squeeze(capsules,axis=1)
            return capsules



    def squash(self,vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return vec_squashed

    def vec_transform(self,input,input_cap_dim,input_cap_num,output_cap_dim,output_cap_num):
        input=tf.expand_dims(input,-1) #[batch,-1,cap_vector,1]
        kernel_size=[1,input_cap_dim,1,output_cap_dim*output_cap_num]
        w_vec_transform=tf.Variable(tf.truncated_normal(shape=kernel_size,mean=0.0,stddev=1.0),name='w_vec_transform')
        u_hat_vec=tf.nn.conv2d(input=input(),filter=w_vec_transform,strides=[1,1,1,1],padding='VALID') #[batch,-1,1,out]
        u_hat_vec=tf.squeeze(u_hat_vec,axis=2)
        u_hat_vec=tf.reshape(u_hat_vec,(-1,input_cap_num,output_cap_num,output_cap_dim))
        u_hat_vec=tf.transpose(u_hat_vec,[0,2,1,3])
        return u_hat_vec


    def dynamic_routing1(self,input,iters):
        #input shape [batch_size,
        b=tf.zeros_like(input[:,:,:,0])
        for i in range(iters):
            c=tf.nn.softmax(b,1)
            out=self.squash(tf.tensordot(c,input,axes=[2,2]))
            if i<iters-1:
                b=b+tf.tensordot(out,input,axes=[2,3])
        return out

    def dynamic_routing2(self,input,iters):
        input_shape=input.get_shape().as_list()
        b=tf.constant(tf.zeros([-1,input_shape[1],output_dim,1,1]))
        w_routing=tf.get_variable('w_routing',shape=[1,input_shape[1],output_cap_dim*output_cap_num]+input_shape[-2:],
                                  dtype=tf.float32,initializer=tf.random_normal_initializer(0.5))
        b_routing=tf.get_variable('b_routing',shape=(1,1,output_cap_dim,output_cap_num,1))
        inpur=tf.tile(input,[1,1,output_cap_dim*output_cap_num,1,1])
        u_hat=tf.reduce_sum(w_routing*input,axis=3,keep_dims=True)
        u_hat=tf.reshape(u_hat,shape=[-1,input_shape[1],output_cap_num,output_cap_dim])
        u_hat_stopped=tf.stop_gradient(u_hat,name='u_hat_stopped')

        for iter in iters:
            with tf.variable_scope('iter_'+str(iter)):
                c=tf.nn.softmax(b,2)
                if iter<iters-1:
                    s=tf.multiply(c,u_hat_stopped)
                    s=tf.reduce_sum(s,axis=1,keep_dims=True)+b_routing
                    v=self.squash(s)

                    v_tile=tf.tile(v,[1,input_shape[1],1,1,1])
                    uv=tf.reduce_sum(u_hat_stopped*v_tile,axis=3,keep_dims=True)
                    b+=uv
                elif iter==iters-1:
                    s=tf.multiply(c,u_hat)


    def em_routing(self):
        pass
