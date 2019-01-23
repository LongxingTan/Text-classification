
import tensorflow as tf
from models._bert_model import *
from models._optimization import *


def _model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                      num_train_steps, num_warmup_steps, use_one_hot_embeddings):
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

        (total_loss, per_example_loss, logits, probabilities) = TextBert(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        # scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        # output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)


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

class TextBert(object):
    def __init__(self):
        pass

    def build(self):
        model = BertModel(bertconfig=bert_config, is_training=is_training, input_ids=input_ids,
                                   input_mask=input_mask, token_type_ids=segment_ids,
                                   use_one_hot_embeddings=use_one_hot_embeddings)
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(name='out_w', shape=[num_labels, hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.2))
        output_bias = tf.get_variable(name='out_b', shape=[num_labels], initializer=tf.zeros_initializer())

        with tf.name_scope('loss'):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.softmax(logits, axis=-1)
            log_prob = tf.nn.log_softmax(logits, axis=-1)

            onehot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(onehot_labels * log_prob, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, per_example_loss, logits, probabilities)

    def __call__(self, *args, **kwargs):
        self.build()
