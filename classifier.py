
import os
import logging
from config import *
from utils import *
import random
import numpy as np
import tensorflow as tf

from model import BertConfig,BertModel


def create_model(bert_config,is_training,input_ids,input_mask,segment_ids,labels,num_labels,use_one_hot_embeddings):
    model=BertModel(config=bert_config,is_training=is_training,input_ids=input_ids,
                    input_mask=input_mask,token_type_ids=segment_ids,use_one_hot_embeddings=use_one_hot_embeddings)
    output_layer=model.get_pooled_output()
    hidden_size=output_layer.shape[-1].value

    output_weights=tf.get_variable(name='out_w',shape=[num_labels,hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.2))
    output_bias=tf.get_variable(name='out_b',shape=[num_labels],initializer=tf.zeros_initializer())

    with tf.name_scope('loss'):
        if is_training:
            output_layer=tf.nn.dropout(output_layer,keep_prob=0.9)

        logits=tf.matmul(output_layer,output_weights,transpose_b=True)
        logits=tf.nn.bias_add(logits,output_bias)

        probabilities=tf.nn.softmax(logits,axis=-1)
        log_prob=tf.nn.log_softmax(logits,axis=-1)

        onehot_labels=tf.onehot(labels,depth=num_labels,dtype=tf.float32)
        per_example_loss=-tf.reduce_sum(onehot_labels*log_prob,axis=-1)
        loss=tf.reduce_mean(per_example_loss)
        return (loss,per_example_loss,logits,probabilities)

def main():
    config = Config()
    bert_config=BertConfig.from_json_file(config.bert_config_file)
    online_processor=OnlineProcessor()

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        raise ValueError("Output directory already exists and is not empty")
    os.makedirs(config.output_dir,exist_ok=True)

    if config.do_train:
        train_examples=online_processor.get_train_examples(config.data_dir)
    label_list=online_processor.get_labels()

    logging.info("label size: {}".format(len(label_list)))

    (loss, per_example_loss, logits, probabilities)=create_model(bert_config=bert_config,is_training=config.do_train,
                                                                 )




