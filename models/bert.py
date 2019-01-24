
import tensorflow as tf
from models._bert_model import *
from models._optimization import *


class TextBert(object):
    def __init__(self,params,train):
        self.params=params
        self.train = train
        print('is training?',self.train)

    def build(self,inputs):
        bert_config=BertConfig.from_json_file(self.params['bert_config_file'])
        model = BertModel(bertconfig=bert_config, is_training=self.train, input_ids=inputs)
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        with tf.name_scope('loss'):
            if self.train:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            self.logits = tf.layers.dense(output_layer, units=self.params['n_class'])


    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits
