import tensorflow as tf
from models._embedding import Embedding_layer


class Bi_LSTM(object):
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.embedding_layer=Embedding_layer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)

    def build(self,inputs):
        with tf.name_scope('embed'):
            embedding_outputs=self.embedding_layer(inputs)

        if self.training:
            embedding_outputs=tf.nn.dropout(embedding_outputs,1.0)

        with tf.name_scope('bi_lstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
            if self.training:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.params['rnn_dropout_keep'])
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.params['rnn_dropout_keep'])

            all_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                             inputs=embedding_outputs,
                                                             sequence_length=None, dtype=tf.float32)
            all_outputs = tf.concat(all_outputs, 2)
            h_outputs = all_outputs[:, -1, :]

        if self.training:
            h_outputs=tf.nn.dropout(h_outputs,self.params['dropout_keep'])

        with tf.name_scope('output'):
            self.logits = tf.layers.dense(h_outputs,units=self.params['n_class'], name="logits")

    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits