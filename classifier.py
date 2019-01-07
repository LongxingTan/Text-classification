# feed data with tf.data
# tf.estimator

import os
import logging
from config import *
from utils import *
import random
import numpy as np
import tensorflow as tf
import collections
import modeling
import optimization

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def create_model(bert_config,is_training,input_ids,input_mask,segment_ids,labels,num_labels,use_one_hot_embeddings):
    model=modeling.BertModel(bertconfig=bert_config,is_training=is_training,input_ids=input_ids,
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






def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op=optimization.create_optimizer(total_loss,learning_rate,num_train_steps,num_warmup_steps)
            output_spec=tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,train_op=train_op)


        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=probabilities)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    # each sample has a corresponding feature dict
    # contains input id: token ids
    # input mask: whether or not value is zero padding
    # segment id: mask showing first part or second part of sequence
    # label value
    # just unpacks different types to a combined list
    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def main():
    bert_config=modeling.BertConfig.from_json_file(config.bert_config_file)
    online_processor=OnlineProcessor()
    tokenizer=tokenization.FullTokenizer(vocab_file=config.vocab_file,do_lower_case=config.do_lower_case)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        raise ValueError("Output directory already exists and is not empty")
    os.makedirs(config.output_dir,exist_ok=True)

    if config.is_training:
        train_examples=online_processor.get_train_examples(config.data_dir)
        num_train_steps=int(len(train_examples)/config.train_batch_size*config.num_train_epochs)
        num_warmup_steps=int(num_train_steps*config.warmup_proportion)
    label_list=online_processor.get_labels()

    logging.info("label size: {}".format(len(label_list)))

    model_fn=model_fn_builder(bert_config=bert_config,num_labels=len(label_list),init_checkpoint=config.init_checkpoint,
                              learning_rate=config.learning_rate,num_train_steps=num_train_steps,
                              num_warmup_steps=num_warmup_steps,use_tpu=config.use_tpu,
                              use_one_hot_embeddings=config.use_tpu)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=None)

    if config.is_training:
        train_file = os.path.join(config.data_dir, "train.tf_record")

        # example to features and then serialise to string and put into file
        label_map_train = file_based_convert_examples_to_features(
            train_examples, label_list, config.max_seq_length, tokenizer, train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=config.max_seq_length,
            is_training=True,
            drop_remainder=True)

        # Start training
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)



if '__main__'==__name__:
    main()
