from models._embedding import Embedding_layer
from model_params import params


class Transformer(object):
    def __init__(self, train):
        self.train = train
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])

    def build(self, inputs):
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits