import tensorflow as tf
from models._embedding import Embedding_layer
from models._normalization import BatchNormalization,LayerNormalization


class SelfAttention(object):
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

        with tf.variable_scope('lstm'):
            fw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.params['rnn_hidden_size'])
            bw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.params['rnn_hidden_size'])
            fw_cell_drop=tf.nn.rnn_cell.DropoutWrapper(fw_cell,input_keep_prob=self.params['rnn_dropout_keep'])
            bw_cell_drop=tf.nn.rnn_cell.DropoutWrapper(bw_cell,input_keep_prob=self.params['rnn_dropout_keep'])
            lstm_outs,_=tf.nn.bidirectional_dynamic_rnn(fw_cell_drop,bw_cell_drop,inputs=embedding_outputs,dtype=tf.float32)
            lstm_encoder=tf.concat(lstm_outs,axis=2)  #shape [Batch_size,sentences_len,2*lstm_hidden_size]

        with tf.variable_scope('attention'):
            attention_w1=tf.Variable(tf.random_normal((2*self.params['rnn_hidden_size'],self.params['attention_hidden_size']),stddev=0.1))
            attention_b1=tf.Variable(tf.random_normal([self.params['attention_hidden_size']],stddev=0.1))

            att=tf.tanh(tf.add(tf.tensordot(lstm_encoder,attention_w1,axes=1),attention_b1)) #shape [batch_size,sentences_len,att_size]
            attention_w2=tf.Variable(tf.random_normal([self.params['attention_hidden_size']],stddev=0.1))

            att_alpha_pre=tf.nn.softmax(tf.tensordot(att,attention_w2,axes=1),name='alpha')  #shape [batch_size,sentences_len]
            att_alpha=tf.expand_dims(att_alpha_pre,-1)
            att_output=tf.reduce_sum(lstm_encoder*att_alpha,1) #shape [batch_size,2*lstm_hidden]

        with tf.variable_scope('penalization'):
            if self.params['penalization']:
                tile_eye=tf.reshape(tf.tile(tf.eye(self.params['seq_length']),[self.params['batch_size'],1]),[-1,self.params['seq_length'],self.params['seq_length']])
                Attention_t=tf.matmul(tf.transpose(att_output,[0,2,1]),att_output)-tile_eye
                self.penalize=tf.square(tf.norm(Attention_t,axis=[-2,-1],ord='fro'))

        with tf.variable_scope('output'):
            logits=tf.layers.dense(att_output,units=self.params['n_class'])
        return logits

    def __call__(self, inputs, targets=None):
        logits=self.build(inputs)
        return logits
