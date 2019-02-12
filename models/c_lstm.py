import tensorflow as tf
from models._embedding import Embedding_layer
from model_params import params


class C_LSTM(object):
    def __init__(self, training):
        self.training = training
        self.embedding_layer = Embedding_layer(vocab_size=params['vocab_size'],
                                               embed_size=params['embedding_size'],
                                               embedding_type=params['embedding_type'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.training:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)


    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits