import tensorflow as tf
from models._embedding import EmbeddingLayer


class TextRNN(object):
    # rnn model, gru or lstm cell, direction or bi-direction, attention or not, pooling or not
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.use_birnn=params['use_birnn']
        self.use_attention=params['use_attention']
        self.use_pooling=params['use_pooling']
        self.embedding_layer=EmbeddingLayer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)

    def build(self,inputs):
        with tf.name_scope('embed'):
            embedding_outputs=self.embedding_layer(inputs)

        if self.training:
            embedding_outputs=tf.nn.dropout(embedding_outputs,keep_prob=self.params['embedding_dropout_keep'])

        with tf.name_scope('rnn'):
            cell = self.create_rnn_cell(num_units=self.params['rnn_hidden_size'],
                                        rnn_cell_type=self.params['use_rnn_cell'],
                                        name='cell')
            if self.training:
                cell= tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.params['rnn_dropout_keep'])
            if self.use_birnn:
                cell_bw=self.create_rnn_cell(num_units=self.params['rnn_hidden_size'],
                                             rnn_cell_type=self.params['use_rnn_cell'],
                                             name='cell_bw')
                if self.training:
                    cell_bw=tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.params['rnn_dropout_keep'])

            sequence_length = self.get_length(inputs)
            if self.use_birnn:
                all_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell_bw,
                                                                 inputs=embedding_outputs,
                                                                 sequence_length=sequence_length,
                                                                 dtype=tf.float32)
                all_outputs = tf.concat(all_outputs, 2)  # [Batch_size,sentences_len,2* lstm_hidden_size]
            else:
                all_outputs, final_states = tf.nn.dynamic_rnn(cell=cell,
                                                              inputs=embedding_outputs,
                                                              sequence_length=sequence_length, dtype=tf.float32)

        if self.use_pooling:
            with tf.name_scope('pool'):
                # rnn_outputs = tf.reduce_max(all_outputs, axis=1)
                max_pool = tf.layers.max_pooling1d(inputs=all_outputs,
                                                   pool_size=self.params['seq_length'],
                                                   strides=1)  # => batch_size * 1 * filters
                avg_pool = tf.layers.average_pooling1d(inputs=all_outputs,
                                                       pool_size=self.params['seq_length'],
                                                       strides=1)
                rnn_outputs = tf.squeeze(tf.concat([max_pool, avg_pool], axis=-1), axis=1)

        if self.use_attention:
            rnn_outputs=self.build_attention(inputs=all_outputs)

            with tf.variable_scope('penalization'):
                if self.params['penalization']:
                    tile_eye = tf.reshape(tf.tile(tf.eye(self.params['seq_length']), [self.params['batch_size'], 1]),
                                          [-1, self.params['seq_length'], self.params['seq_length']])
                    Attention_t = tf.matmul(tf.transpose(rnn_outputs, [0, 2, 1]), rnn_outputs) - tile_eye
                    self.penalize = tf.square(tf.norm(Attention_t, axis=[-2, -1], ord='fro'))

        if (not self.use_pooling) and (not self.use_attention):
            #rnn_outputs = self.last_relevant(all_outputs, sequence_length)
            #rnn_outputs = final_states.all_outputs
            rnn_outputs=all_outputs[:,-1,:]

        rnn_outputs = tf.layers.batch_normalization(rnn_outputs,training=self.training)
        if self.training:
            rnn_outputs=tf.nn.dropout(rnn_outputs,keep_prob=self.params['dropout_keep'])
        with tf.name_scope('output'):
            logits = tf.layers.dense(rnn_outputs,units=self.params['n_class'],name="logit")
        return logits

    def __call__(self,inputs,targets=None):
        logits=self.build(inputs)
        return logits

    def create_rnn_cell(self,num_units,rnn_cell_type,name=None):
        with tf.name_scope(name):
            if rnn_cell_type=='lstm':
                cell=tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                                             activation=tf.nn.tanh,
                                             state_is_tuple=True)
            elif rnn_cell_type=='gru':
                cell=tf.nn.rnn_cell.GRUCell(num_units=num_units,
                                            activation=tf.nn.tanh)
            if self.training:
                cell=tf.contrib.rnn.DropoutWrapper(cell=cell,input_keep_prob=self.params['rnn_dropout_keep'])
        return cell

    @staticmethod
    def get_length(seq):
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

    def build_attention(self,inputs):
        with tf.variable_scope('attention'):
            attention_w1 = tf.Variable(
                tf.random_normal((2 * self.params['rnn_hidden_size'], self.params['attention_hidden_size']),
                                 stddev=0.1))
            attention_b1 = tf.Variable(tf.random_normal([self.params['attention_hidden_size']], stddev=0.1))

            att = tf.tanh(tf.add(tf.tensordot(inputs, attention_w1, axes=1),
                                 attention_b1))  # shape [batch_size,sentences_len,att_size]
            attention_w2 = tf.Variable(tf.random_normal([self.params['attention_hidden_size']], stddev=0.1))

            att_alpha_pre = tf.nn.softmax(tf.tensordot(att, attention_w2, axes=1),
                                          name='alpha')  # shape [batch_size,sentences_len]
            att_alpha = tf.expand_dims(att_alpha_pre, -1)
            rnn_outputs = tf.reduce_sum(inputs * att_alpha, 1)  # shape [batch_size,2*lstm_hidden]
        return rnn_outputs

    def build_attention2(self,inputs,):
        with tf.variable_scope('attention'):
            _atn_in = tf.expand_dims(inputs, axis=2)
            atn_w = tf.Variable(
                tf.truncated_normal(shape=[1, 1, 2 * self.params['rnn_hidden_size'], self.params['attention_hidden_size']],
                                    stddev=0.1), name='atn_w')
            atn_b = tf.Variable(tf.zeros(shape=[self.params['attention_hidden_size']]))
            atn_v = tf.Variable(tf.truncated_normal(shape=[1, 1, self.params['attention_hidden_size'], 1], stddev=0.1),
                                     name='atn_v')
            atn_activation = tf.nn.tanh(
                tf.nn.conv2d(_atn_in, atn_w, strides=[1, 1, 1, 1], padding='SAME') + atn_b)
            atn_scores = tf.nn.conv2d(atn_activation, atn_v, strides=[1, 1, 1, 1], padding='SAME')
            atn_probs = tf.nn.softmax(tf.squeeze(atn_scores, [2, 3]))
            _atn_out = tf.matmul(tf.expand_dims(atn_probs, 1), inputs)
            attention_output = tf.squeeze(_atn_out, [1], name='atn_out')
        return attention_output
