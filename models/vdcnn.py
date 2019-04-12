import tensorflow as tf
from models._embedding import Embedding_layer
from models._normalization import BatchNormalization,LayerNormalization
from models._k_max_pooling import KMaxPooling


class VDCNN(object):
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.embedding_layer=Embedding_layer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)
        self.bn_layer=BatchNormalization()
        self.kmax_pooling=KMaxPooling()
        self.depth=29
        self.num_conv_blocks = (2, 2, 2, 2)

    def build(self,inputs):
        with tf.variable_scope("embed"):
            embedded_outputs=self.embedding_layer(inputs) # => batch_size* seq_length* embedding_dim
        if self.training:
            embedded_outputs=tf.nn.dropout(embedded_outputs,self.params['embedding_dropout_keep'])

        with tf.variable_scope("conv_1st_layer" ):
            conv1 = tf.layers.conv1d(inputs=embedded_outputs,
                                     filters=self.params['filters'],
                                     kernel_size=[3],
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu)  # => batch_size *(seq_length-kernel_size+1)* filters
            out=conv1


        for i,layer in enumerate(self.num_conv_blocks):
            with tf.variable_scope('conv_block_%s' % i):
                for j in range(layer):
                    with tf.variable_scope('sub_%s' %j):
                        out = self.identity_block(out, filters=128, kernel_size=3,name=str(i)+str(j))
                        out=tf.layers.batch_normalization(out,center=True,scale=True,momentum=0.99,training=self.training,name=str(i)+str(j))
                        out = self.conv_block(out, filters=128, kernel_size=3, pool_type='max',name=str(i)+str(j))


        #shape=out.get_shape().as_list()#Todo
        #print(out.get_shape().as_list())
        #out=tf.layers.max_pooling1d(out,pool_size=[shape[1]],strides=1,name='max_pool')
        self.cnn_out = tf.squeeze(out, axis=1)
        self.cnn_out = self.bn_layer(self.cnn_out)

        if self.training:
            self.cnn_out = tf.nn.dropout(self.cnn_out, self.params['dropout_keep'])
        self.logits = tf.layers.dense(self.cnn_out, units=self.params['n_class'])



    def conv_block(self,inputs,filters,kernel_size,use_bias=False,shortcut=False,pool_type='max', sorted=True,name=1):
        with tf.variable_scope('conv_block_%s' % name):
            block_conv1 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=1,
                                           padding='same', name='conv_block1')
            block_bn1 = self.bn_layer(block_conv1,name='conv_bn1')
            outputs = tf.nn.relu(block_bn1)


            if shortcut:
                residual = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=1, strides=2, padding='same')
                residual = self.bn_layer(residual)
                outputs = self.downsampling(outputs, pool_type=pool_type, sorted=sorted,name=name)
                outputs= tf.add(outputs, residual)
                outputs = tf.nn.relu(outputs)
            else:
                outputs = self.downsampling(outputs, pool_type=pool_type, sorted=sorted,name=name)
            if pool_type is not None:
                outputs = tf.layers.conv1d(outputs, filters=2 * filters, kernel_size=1, strides=1,
                                       padding='same',name='conv_block3')
                outputs = self.bn_layer(outputs,name='conv_bn3')
        return outputs


    def identity_block(self,inputs,filters,kernel_size,use_bias=False,shortcut=False,name=1):
        with tf.variable_scope('identity_block_%s' % name):
            block_conv1 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=1,
                                           padding='same', name='identity_conv1')
            outputs= self.bn_layer(block_conv1,name='identity_bn1')
            if shortcut:
                out = tf.add(outputs, inputs)
        return tf.nn.relu(outputs)

    def downsampling(self,inputs,pool_type,sorted,name=1):
        with tf.variable_scope('down_sampling_%s' % name):
            if pool_type == 'max':
                outputs = tf.layers.max_pooling1d(inputs, pool_size=[3], strides=2, padding='same', name='down_pool_max')
            elif pool_type == 'k_max':
                outputs = self.kmax_pooling(inputs,name=name)
            elif pool_type == 'conv':
                outputs = tf.layers.conv1d(inputs, filters=2 * 256, kernel_size=3, strides=2, padding='same',
                                       name='down_conv_pool')
                outputs = self.bn_layer(outputs)
            elif pool_type is None:
                outputs = inputs
            else:
                raise ValueError("Unsupported pooling type: %s" % pool_type)
        return outputs

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits
