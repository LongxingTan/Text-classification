import tensorflow as tf
from models._embedding import Embedding_layer
from model_params import params


class RCNN(object):
    def __init__(self, train):
        self.train = train
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.train:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits