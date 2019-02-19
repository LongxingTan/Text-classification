from models._embedding import Embedding_layer


class Transformer(object):
    def __init__(self, training,params):
        self.training = training
        self.params=params
        self.embedding_layer = Embedding_layer(vocab_size=params['vocab_size'],
                                               embed_size=params['embedding_size'],
                                               embedding_type=params['embedding_type'],
                                               params=params)
    def build(self, inputs):
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits