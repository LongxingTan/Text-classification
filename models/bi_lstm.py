import tensorflow as tf
from models._embedding import Embedding_layer
from models._normalization import BatchNormalization,LayerNormalization

class Bi_LSTM(object):
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.embedding_layer=Embedding_layer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)
        self.bn_layer = BatchNormalization()


    def build(self,inputs):
        with tf.name_scope('embed'):
            embedding_outputs=self.embedding_layer(inputs)

        if self.training:
            embedding_outputs=tf.nn.dropout(embedding_outputs,keep_prob=self.params['embedding_dropout_keep'])

        with tf.name_scope('bi_lstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
            if self.training:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.params['rnn_dropout_keep'])
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.params['rnn_dropout_keep'])

            all_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                             inputs=embedding_outputs,
                                                             sequence_length=None,
                                                             dtype=tf.float32)
            all_outputs = tf.concat(all_outputs, 2)  # [Batch_size,sentences_len,2* lstm_hidden_size]

        with tf.name_scope('pool'):
            #rnn_outputs = tf.reduce_max(all_outputs, axis=1)
            max_pool=tf.layers.max_pooling1d(inputs=all_outputs,
                                              pool_size=self.params['seq_length'],
                                              strides=1) # => batch_size * 1 * filters
            avg_pool=tf.layers.average_pooling1d(inputs=all_outputs,
                                                 pool_size=self.params['seq_length'],
                                                 strides=1)
            rnn_outputs=tf.squeeze(tf.concat([max_pool,avg_pool],axis=-1),axis=1)
            rnn_outputs = self.bn_layer(rnn_outputs)

        if self.training:
            rnn_outputs=tf.nn.dropout(rnn_outputs,keep_prob=self.params['dropout_keep'])

        with tf.name_scope('output'):
            self.logits = tf.layers.dense(rnn_outputs,units=self.params['n_class'],name="logit")

    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits
