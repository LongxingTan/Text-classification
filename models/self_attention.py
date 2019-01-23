import tensorflow as tf
from models._embedding import Embedding_layer
from models._model_params import *


class Self_attention(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.train:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)

        with tf.variable_scope('lstm'):
            fw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden)
            bw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden)
            fw_cell_drop=tf.nn.rnn_cell.DropoutWrapper(fw_cell,input_keep_prob=self.dropout_keep_prob)
            bw_cell_drop=tf.nn.rnn_cell.DropoutWrapper(bw_cell,input_keep_prob=self.dropout_keep_prob)
            lstm_out,_=tf.nn.bidirectional_dynamic_rnn(fw_cell_drop,bw_cell_drop,inputs=embed_encoder,dtype=tf.float32)
            lstm_encoder=tf.concat([lstm_out[0],lstm_out[1]],axis=2)  #shape [Batch_size,n_sentences,2*lstm_hidden_size]

        with tf.variable_scope('attention'):
            attention_w1=tf.Variable(tf.random_normal((2*self.lstm_hidden,self.att_dim),stddev=0.1))
            attention_b1=tf.Variable(tf.random_normal([self.att_dim],stddev=0.1))

            att=tf.tanh(tf.add(tf.tensordot(lstm_encoder,attention_w1,axes=1),attention_b1)) #shape [batch_size,n_sentences,att_size]
            attention_w2=tf.Variable(tf.random_normal([self.att_dim],stddev=0.1))

            att_alpha_pre=tf.nn.softmax(tf.tensordot(att,attention_w2,axes=1),name='alpha')  #shape [batch_size,n_sentences]
            att_alpha=tf.expand_dims(att_alpha_pre,-1)
            att_output=tf.reduce_sum(lstm_encoder*att_alpha,1) #shape [batch_size,2*lstm_hidden]

        with tf.variable_scope('penalization'):
            if self.penalization:
                tile_eye=tf.reshape(tf.tile(tf.eye(self.sentence_length),[self.batch_size,1]),[-1,self.sentence_length,self.sentence_length])
                Attention_t=tf.matmul(tf.transpose(att_output,[0,2,1]),att_output)-tile_eye
                self.penalize=tf.square(tf.norm(Attention_t,axis=[-2,-1],ord='fro'))

        with tf.variable_scope('output'):
            self.logits=tf.layers.dense(att_output,units=params['n_class'])

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits