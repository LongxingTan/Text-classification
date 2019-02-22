import tensorflow as tf
from models._embedding import Embedding_layer
from models._layer_normalization import BatchNormalization,LayerNormalization


class TextCNN(object):
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.embedding_layer=Embedding_layer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)
        self.bn_layer=BatchNormalization()

    def build(self,inputs):
        with tf.name_scope("embed"):
            embedded_outputs=self.embedding_layer(inputs)

        if self.training:
            embedded_outputs=tf.nn.dropout(embedded_outputs,self.params['embedding_dropout_keep'])

        conv_output = []
        for i, kernel_size in enumerate(self.params['kernel_sizes']):
            with tf.name_scope("conv_%s" % kernel_size):
                conv1=tf.layers.conv1d(inputs=embedded_outputs,
                                       filters=self.params['filters'],
                                       kernel_size=[kernel_size],
                                       strides=1,
                                       padding='valid',
                                       activation=tf.nn.relu)
                pool1=tf.layers.max_pooling1d(inputs=conv1,
                                              pool_size=self.params['seq_length'] - kernel_size + 1,
                                              strides=1)
                conv_output.append(pool1)

        self.cnn_output_concat=tf.concat(conv_output,2) #[batch_size,1,params['filters']*len(params['kernel_sizes'])
        self.cnn_out=tf.squeeze(self.cnn_output_concat,axis=1)
        self.cnn_out = self.bn_layer(self.cnn_out)

        if self.training:
            self.cnn_out=tf.nn.dropout(self.cnn_out,self.params['dropout_keep'])
        self.logits=tf.layers.dense(self.cnn_out,units=self.params['n_class'])


    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits
