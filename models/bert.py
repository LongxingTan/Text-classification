from models._bert_model import *


class TextBert(object):
    def __init__(self,training,params):
        self.training = training
        self.params=params
        assert self.params['chinese_seg']=='char',"Please set the chinese_seg to 'char' while run Bert model"

    def build(self,inputs):
        bert_config=BertConfig.from_json_file(self.params['bert_config_file'])
        model = BertModel(bertconfig=bert_config, is_training=self.training, input_ids=inputs)
        output_layer = model.get_pooled_output()  # [None, 768]

        with tf.name_scope('output'):
            if self.training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=self.params['dropout_keep'])
            logits = tf.layers.dense(output_layer, units=self.params['n_class'])
        return logits

    def __call__(self,inputs,targets=None):
        logits=self.build(inputs)
        return logits
