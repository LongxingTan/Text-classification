from models._bert_model import *
from model_params import params


class TextBert(object):
    def __init__(self,training):
        self.training = training

    def build(self,inputs):
        bert_config=BertConfig.from_json_file(params['bert_config_file'])
        model = BertModel(bertconfig=bert_config, is_training=self.training, input_ids=inputs)
        output_layer = model.get_pooled_output()

        with tf.name_scope('loss'):
            if self.training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            self.logits = tf.layers.dense(output_layer, units=params['n_class'])


    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits
