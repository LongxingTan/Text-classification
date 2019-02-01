import tensorflow as tf
from models._embedding import Embedding_layer
from models._model_params import params


class TextCNN(object):
    def __init__(self,training):
        self.training=training
        self.embedding_layer=Embedding_layer(params['vocab_size'],params['embedding_size'])

    def build(self,inputs):
        with tf.name_scope("embed"):
            embedded_outputs=self.embedding_layer(inputs) #[batch_size,max_sentence_length,embedding_size]

        if self.training:
            embedded_outputs=tf.nn.dropout(embedded_outputs,1.0)

        self.embedding_chars_expanded = tf.expand_dims(embedded_outputs, -1)
        pooled_output = []
        for i, filter_size in enumerate(params['kernel_sizes']):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_chars_expanded.get_shape().as_list()[2], 1,params['filters']]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[params['filters']]), name='b')
                conv = tf.nn.conv2d(self.embedding_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID',
                                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # pooling filter depends on the padding style
                pooled = tf.nn.max_pool(h, ksize=[1, params['seq_length']  - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_output.append(pooled)

        num_filters_total = params['filters'] * len(params['kernel_sizes'])
        self.h_pool = tf.concat(pooled_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])



        with tf.name_scope("output"):
            W_out = tf.get_variable("W_out", shape=[num_filters_total, params['n_class']],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[params['n_class']]), name='b')

            self.logits = tf.nn.xw_plus_b(self.h_pool_flat, W_out, b, name='scores')

        '''
        conv_output = []
        for i, kernel_size in enumerate(params['kernel_sizes']):
            with tf.name_scope("conv_maxpool_%s" % kernel_size):
                conv1=tf.layers.conv1d(inputs=embedded_outputs,
                                       filters=params['filters'],
                                       kernel_size=[kernel_size],
                                       strides=1,
                                       padding='valid',
                                       activation=tf.nn.relu)
                pool1=tf.layers.max_pooling1d(inputs=conv1,
                                              pool_size=params['seq_length'] - kernel_size + 1,
                                              strides=1)
                conv_output.append(pool1)

        self.cnn_output_concat=tf.concat(conv_output,2) #[batch_size,1,params['filters']*len(params['kernel_sizes'])
        self.cnn_out=tf.squeeze(self.cnn_output_concat,axis=1)
        #print(self.cnn_out.get_shape().as_list())

        if self.training:
            self.cnn_out=tf.nn.dropout(self.cnn_out,1.0)

        self.logits=tf.layers.dense(self.cnn_out,units=self.params['n_class'])
        '''



    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits