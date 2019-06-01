import tensorflow as tf
from models._embedding import Embedding_layer


class RCNN(object):
    def __init__(self, training,params):
        self.training = training
        self.params=params
        self.embedding_layer = Embedding_layer(vocab_size=params['vocab_size'],
                                               embed_size=params['embedding_size'],
                                               embedding_type=params['embedding_type'],
                                               params=params)

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.training:
            embedding_outputs = tf.nn.dropout(embedding_outputs, self.params['embedding_dropout_keep'])

        with tf.variable_scope('bi-rnn'):
            fw_cell=tf.nn.rnn_cell.LSTMCell(self.params['rnn_hidden_size'])
            fw_cell=tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=1.0)
            bw_cell=tf.nn.rnn_cell.LSTMCell(self.params['rnn_hidden_size'])
            bw_cell=tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=1.0)
            (outputs_fw,outputs_bw),_=tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                      cell_bw=bw_cell,
                                                                      inputs=embedding_outputs,
                                                                      dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(outputs_fw)[0], 1, tf.shape(outputs_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape), outputs_fw[:, :-1]], axis=1, name="context_left")
            self.c_right = tf.concat([outputs_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            self.x = tf.concat([self.c_left, embedding_outputs, self.c_right], axis=2, name="x")
            embedding_size = 2 * self.params['rnn_hidden_size'] + self.params['embedding_size']

        with tf.name_scope("text-representation"):
            W2 = tf.Variable(tf.random_uniform([embedding_size, self.params['dense_hidden_size']], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.params['dense_hidden_size']]), name="b2")
            self.y2 = tf.einsum('aij,jk->aik', self.x, W2) + b2

        with tf.name_scope("max-pooling"):
            y3 = tf.reduce_max(self.y2, axis=1)

        with tf.variable_scope('output'):
            logits=tf.layers.dense(y3,units=self.params['n_class'])
        return logits

    def __call__(self, inputs, targets=None):
        logits=self.build(inputs)
        return logits
