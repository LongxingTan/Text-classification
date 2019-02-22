import tensorflow as tf
from models._embedding import Embedding_layer
from models._layer_normalization import BatchNormalization,LayerNormalization

class LSTM(object):
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.embedding_layer=Embedding_layer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)
        self.bn_layer = BatchNormalization()


    def build(self,inputs):
        self.text_length = self._length(inputs)
        with tf.name_scope('embed'):
            embedding_outputs=self.embedding_layer(inputs)

        if self.training:
            embedding_outputs=tf.nn.dropout(embedding_outputs,keep_prob=self.params['embedding_dropout_keep'])

        with tf.name_scope('bi_lstm'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['lstm_hidden_size'])
            if self.training:
                cell= tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.params['rnn_dropout_keep'])

            all_outputs, final_states = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=embedding_outputs,
                                               sequence_length=self.text_length, dtype=tf.float32)
            #rnn_outputs = self.last_relevant(all_outputs, self.text_length)
            #rnn_outputs = final_states.all_outputs
            rnn_outputs=all_outputs[:,-1,:]
            rnn_outputs = self.bn_layer(rnn_outputs)

        if self.training:
            rnn_outputs=tf.nn.dropout(rnn_outputs,keep_prob=self.params['dropout_keep'])

        with tf.name_scope('output'):
            self.logits = tf.layers.dense(rnn_outputs,units=self.params['n_class'],name="logit")

    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits

    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

    def extract_axis_1(data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res