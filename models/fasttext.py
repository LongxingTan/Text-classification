from models._embedding import Embedding_layer
from model_params import params


class FastText(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])

    def build(self, inputs):
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits