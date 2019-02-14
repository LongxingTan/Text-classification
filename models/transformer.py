from models._embedding import Embedding_layer
from model_params import params


class Transformer(object):
    def __init__(self, training):
        self.training = training
        self.embedding_layer = Embedding_layer(vocab_size=params['vocab_size'],
                                               embed_size=params['embedding_size'],
                                               embedding_type=params['embedding_type'])

    def build(self, inputs):
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits