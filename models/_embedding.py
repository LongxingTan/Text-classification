import tensorflow as tf
import numpy as np
import gensim

class Embedding_layer():
    def __init__(self,vocab_size,embed_size):
        super(Embedding_layer,self).__init__()
        self.vocab_size=vocab_size
        self.embed_size=embed_size

    def build(self,_):
        pass

    def __call__(self,x,embedding_type='random',embedding_file='/',vocab_file='/'):

        if embedding_type=='random':
            self.embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),
                                               name='embedding_w')
            with tf.name_scope("embedding"):
                embeddings = tf.nn.embedding_lookup(self.embedding_table, x)
                return embeddings

        elif embedding_type=='word2vec_finetune':
            embedding_vocab=gensim.models.KeyedVectors.load_word2vec_format(embedding_file,binary=True,
                                                                            encoding='utf-8',unicode_errors='ignore')
            embedding_table=np.zeros((self.vocab_size,self.embed_size))
            for word,i in vocab_file.items():
                if word in embedding_vocab.vocab:
                    embedding_table[i]=embedding_vocab[word]
                else:
                    embedding_table[i]=np.random.random(self.embed_size)
            self.embedding_table=tf.get_variable(name='embedding_w',shape=[self.vocab_size,self.embed_size],
                                                 initializer=tf.constant_initializer(embedding_table),trainable=True)
            embeddings=tf.nn.embedding_lookup(self.embedding_table,x)
            return embeddings









