import re
import os
import bz2,gzip
import numpy as np
import tensorflow as tf
import gensim
import tokenization
from collections import Counter


class Embedding_layer():
    def __init__(self,vocab_size,embed_size,params,embedding_type='random',vocab=None):
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.embedding_type=embedding_type
        self.vocab=vocab
        self.params=params

    def create_embedding_table(self,embedding_type,name):
        oov_vocab,in_vocab=set(),set()
        if embedding_type=='random':
            embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),name='embed_w')
            return embedding_table

        elif re.search('word2vec',embedding_type) is not None:
            embedding_file=self.params['word2vec_file'] ##https://github.com/Embedding/Chinese-Word-Vectors
            embedding_vocab=gensim.models.KeyedVectors.load_word2vec_format(embedding_file,binary=True,
                                                                            encoding='utf-8',unicode_errors='ignore')
            embedding_table=np.zeros((self.vocab_size,self.embed_size))
            self.vocab,index_vocab=tokenization.load_vocab(vocab_file=os.path.join(self.params['data_dir'],'vocab_word.txt'),
                                                           params=self.params)

            for word,i in self.vocab.items():
                if word in embedding_vocab.vocab:
                    embedding_table[i]=embedding_vocab[word]
                    in_vocab.add(word)
                else:
                    embedding_table[i]=np.random.random(self.embed_size)
                    oov_vocab.add(word)
            tf.logging.info('OOV:%f' % (len(oov_vocab)/(len(oov_vocab)+len(in_vocab))))

            if embedding_type=='word2vec_finetune':
                trainable=True
            elif embedding_type=='word2vec_static':
                trainable=False
            else:
                trainable=False
                print("word2vec word embedding type please choose 'static' or 'finetune'")
            embedding_table2 = tf.get_variable(name='embedding_w', shape=[self.vocab_size, self.embed_size],
                                               initializer=tf.constant_initializer(embedding_table), trainable=trainable)
            return embedding_table2

        elif re.search('fasttext',embedding_type) is not None:
            #https://fasttext.cc/docs/en/crawl-vectors.html
            embedding_vocab=self._load_embedding_pretrained(embedding_file=self.params['fasttext_file'])
            embedding_table = np.zeros((self.vocab_size, self.embed_size))
            self.vocab, index_vocab = tokenization.load_vocab(vocab_file=os.path.join(self.params['data_dir'], 'vocab_word.txt'),
                                                              params=self.params)

            for word, i in self.vocab.items():
                if word in embedding_vocab.keys():
                    embedding_table[i]=embedding_vocab[word]
                    in_vocab.add(word)
                else:
                    embedding_table[i]= np.random.random(self.embed_size)
                    oov_vocab.add(word)

            if embedding_type == 'fasttext_finetune':
                trainable = True
            elif embedding_type == 'fasttext_static':
                trainable = False
            else:
                trainable=False
                print("fasttext word embedding type please choose 'static' or 'finetune'")
            embedding_table2 = tf.get_variable(name=name+'embedding_w', shape=[self.vocab_size, self.embed_size],
                                               initializer=tf.constant_initializer(embedding_table),
                                               trainable=trainable)
            tf.logging.info('OOV:%f' % (len(oov_vocab) / (len(oov_vocab) + len(in_vocab))))
            return embedding_table2

        elif re.search('glove',embedding_type) is not None:
            pass

        elif re.search('elmo', embedding_type) is not None:
            print('Invalid embedding type: %s'%self.params['embedding_type'])
            print('elmo please refer to github repository: HIT-SCIR/ELMoForManyLangs')


    def __call__(self,x):
        if self.embedding_type!='multi_channel':
            embedding_table = self.create_embedding_table(embedding_type=self.embedding_type,name=self.embedding_type+'w')
            with tf.name_scope("embedding"):
                embeddings = tf.nn.embedding_lookup(embedding_table, x)
                return embeddings
        else:
            embedding_table1=self.create_embedding_table(embedding_type='word2vec_finetune',name='word2vec_w')
            embeddings1=tf.nn.embedding_lookup(embedding_table1,x)

            embedding_table2=self.create_embedding_table(embedding_type='fasttext_finetune',name='fasttext_w')
            embeddings2=tf.nn.embedding_lookup(embedding_table2,x)
            embeddings=tf.concat([embeddings1,embeddings2],axis=-1)
            return embeddings

    def _load_embedding_pretrained(self,embedding_file):
        data = {}
        if re.search('gz',embedding_file) is not None:
            with gzip.open(embedding_file, mode='rb') as f:
                for line_b in f:
                    line = line_b.decode('utf-8')
                    tokens = line.rstrip().split(' ')
                    data[tokens[0]] = list(map(float, tokens[1:]))
        elif re.search('bz',embedding_file) is not None:
            with bz2.open(embedding_file, mode='rb') as f:
                for line_b in f:
                    line = line_b.decode('utf-8')
                    tokens = line.rstrip().split(' ')
                    data[tokens[0]] = list(map(float, tokens[1:]))
        return data

    def _filter_frequency_vocab(self,x,low=0.05,high=0.95):
        filtered_vocab=[]
        for example in x:
            filtered_vocab.extend(example)
        vocab_count_dict=Counter(filtered_vocab)
        sorted(vocab_count_dict.items(),key=lambda x: x[1],reverse=True)
        value=np.array([value for value in vocab_count_dict.values()])
        value_low_quantile=np.quantile(value,low)
        value_high_quantile=np.quantile(value,high)
        vocab_count_dict_filtered={k:v for k,v in vocab_count_dict.items() if v>=value_low_quantile and v<=value_high_quantile}
        vocab_filtered=list(vocab_count_dict_filtered.keys())
        return vocab_filtered

    def _create_tfidf_feature_dict(self,chi_k): # exists in old version of model_tf_archives/_utils
        pass
