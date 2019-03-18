import tensorflow as tf
from models._embedding import Embedding_layer
from models._normalization import BatchNormalization,LayerNormalization

class HAN(object):
    def __init__(self, training,params):
        self.training = training
        self.params=params
        self.embedding_layer = Embedding_layer(vocab_size=params['vocab_size'],
                                               embed_size=params['embedding_size'],
                                               embedding_type=params['embedding_type'],
                                               params=params)

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs) #[batch_size,seq_length,embedding_dim]

        if self.training:
            embedding_outputs = tf.nn.dropout(embedding_outputs, self.params['embedding_dropout_keep'])

        word_inputs = tf.reshape(embedding_outputs , [-1, each_sentence_length,self.params['embedding_dim']])
        with tf.variable_scope('word_encoder') as scope:
            word_outputs, _ = self.bi_gru_encode(word_inputs, self.params['lstm_hidden_size'], scope)

            with tf.name_scope('attention'):
                word_attention = self.attention(word_outputs, attention_dim=self.params['attention_hidden_size'])
                word_attention = tf.reshape(word_attention, [-1, n_sentences, 2 * self.params['lstm_hidden_size']])

            with tf.name_scope('dropout'):
                word_attention = tf.nn.dropout(word_attention, 1.0)

        with tf.variable_scope('sentence_encoder') as scope:
            sentence_outputs, _ = self.bi_gru_encode(word_attention, self.params['lstm_hidden_size'], scope)
            sentence_outputs.set_shape([None, n_sentences, 2 * self.params['lstm_hidden_size']])

            with tf.name_scope('attention'):
                sentence_attention = self.attention(sentence_outputs, attention_dim=self.params['attention_hidden_size'])

            with tf.name_scope('dropout'):
                sentence_attention = tf.nn.dropout(sentence_attention,1.0)

        with tf.variable_scope('output'):
            self.logits=tf.layers.dense(sentence_attention,units=self.params['n_class'])


    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits

    def bi_gru_encode(self, inputs, lstm_size,scope=None):
        with tf.variable_scope(scope or 'bi_gru'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1.)
            cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_size)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1.)

            enc_out,(enc_state_fw,enc_state_bw)=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,
                                                                                inputs=inputs,sequence_length=None,
                                                                                dtype=tf.float32,scope='L')
            enc_state=tf.concat([enc_state_bw,enc_state_bw],1)
            enc_outputs=tf.concat(enc_out,2)
        return enc_outputs,enc_state

    def attention(self,inputs,attention_dim):
        with tf.name_scope('attention__'):
            attention_vector=tf.get_variable(shape=[attention_dim],dtype=tf.float32,name='attention_vector')
            input_projection=tf.layers.dense(inputs,attention_dim,activation_fn=tf.tanh)
            vector_att=tf.reduce_sum(tf.multiply(input_projection,attention_vector),axis=2,keep_dims=True)
            attention_weights=tf.nn.softmax(vector_att,dim=1)
            weighted_projection=tf.multiply(inputs,attention_weights)
            attention_outputs=tf.reduce_sum(weighted_projection,axis=1)
        return attention_outputs