import tensorflow as tf
from models._embedding import Embedding_layer
from models._model_params import *


class TextCNN(object):
    def __init__(self,params,train):
        self.train=train
        self.params=params
        self.embedding_layer=Embedding_layer(params['vocab_size'],params['embedding_size'])

    def build(self,inputs):
        with tf.name_scope("embed"):
            embedded_outputs=self.embedding_layer(inputs) #[batch_size,max_sentence_length,embedding_size]

        if self.train:
            embedded_outputs=tf.nn.dropout(embedded_outputs,0.9)

        conv_output = []
        for i, kernel_size in enumerate(self.params['kernel_sizes']):
            with tf.name_scope("conv_maxpool_%s" % kernel_size):
                conv1=tf.layers.conv1d(inputs=embedded_outputs,
                                       filters=params['filters'],
                                       kernel_size=[kernel_size],
                                       strides=1,
                                       padding='valid',
                                       activation=tf.nn.relu)
                pool1=tf.layers.max_pooling1d(inputs=conv1,
                                              pool_size=self.params['seq_length'] - kernel_size + 1,
                                              strides=1)
                conv_output.append(pool1)

        num_filters_total=self.params['filters']*len(self.params['kernel_sizes'])
        self.cnn_output_concat=tf.concat(conv_output,2)
        self.cnn_out=tf.reshape(self.cnn_output_concat,[-1,num_filters_total])

        if self.train:
            self.cnn_out=tf.nn.dropout(self.cnn_out,0.9)

        self.logits=tf.layers.dense(self.cnn_out,units=self.params['n_class'])

    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits