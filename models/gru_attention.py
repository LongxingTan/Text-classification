import tensorflow as tf
from models._embedding import Embedding_layer
from models._model_params import *


class GRU_Attention(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.train:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)

        encoder_fw = tf.nn.rnn_cell.GRUCell(self.params['gru_hidden_size'])
        encoder_bw = tf.nn.rnn_cell.GRUCell(self.params['gru_hidden_size'])
        self.encoder_fw = tf.nn.rnn_cell.DropoutWrapper(encoder_fw, input_keep_prob=0.9)
        self.encoder_bw = tf.nn.rnn_cell.DropoutWrapper(encoder_bw, input_keep_prob=0.9)

        with tf.variable_scope('bi_gru') as scope:
            gru_outputs, gru_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw, cell_bw=self.encoder_bw,
                                                                      inputs=self.encoder_input_embeded,
                                                                      dtype=tf.float32)
            self.encoder_output = tf.concat(gru_outputs, 2)
            self.encoder_state = tf.concat(gru_states, 1)

        with tf.variable_scope('attention') as scope:
            self._atn_in = tf.expand_dims(self.encoder_output, axis=2)
            self.atn_w = tf.Variable(
                tf.truncated_normal(shape=[1, 1, 2 * self.gru_hidden_size, self.attention_hidden_size],
                                    stddev=0.1), name='atn_w')
            self.atn_b = tf.Variable(tf.zeros(shape=[self.attention_hidden_size]))
            self.atn_v = tf.Variable(tf.truncated_normal(shape=[1, 1, self.attention_hidden_size, 1], stddev=0.1),
                                     name='atn_v')
            self.atn_activation = tf.nn.tanh(
                tf.nn.conv2d(self._atn_in, self.atn_w, strides=[1, 1, 1, 1], padding='SAME') + self.atn_b)
            self.atn_scores = tf.nn.conv2d(self.atn_activation, self.atn_v, strides=[1, 1, 1, 1], padding='SAME')
            atn_probs = tf.nn.softmax(tf.squeeze(self.atn_scores, [2, 3]))
            _atn_out = tf.matmul(tf.expand_dims(atn_probs, 1), self.encoder_output)
            self.attention_output = tf.squeeze(_atn_out, [1], name='atn_out')

        with tf.variable_scope('output'):
            self.logits=tf.layers.dense(self.attention_output,units=params['n_class'])


    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits

