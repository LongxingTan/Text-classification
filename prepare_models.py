import tensorflow as tf
import logging
from models._loss import create_loss
from models._optimization import create_optimizer_basic_adam,create_optimizer_warmup_adam
from models._eval import create_eval,create_eval_sk
from models.bert import TextBert,get_assignment_map_from_checkpoint
from models.rnn import TextRNN
from models.cnn import TextCNN
from models.capsule import TextCapsule
from models.c_lstm import CLSTM
from models.rcnn import RCNN
from models.vdcnn import VDCNN


def model_fn_builder(textmodel,params,init_checkpoint=None):
    textmodel = eval(textmodel)
    def model_fn(features, labels, mode):
        inputs, targets = features['input_ids'], features['label_ids']
        model=textmodel(training=(mode==tf.estimator.ModeKeys.TRAIN),params=params)
        logits = model(inputs, targets)
        targets_onehot = tf.one_hot(targets, depth=params['n_class'],dtype=tf.float32)

        output_loss = create_loss(logits=logits, y_onehot=targets_onehot, loss_type='cross_entropy')
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = output_loss + reg_loss if reg_loss else output_loss
        prediction_label = tf.argmax(logits, -1,output_type=tf.int32)
        probabilities = tf.reduce_max(tf.nn.softmax(logits, name='softmax_tensor'),axis=-1)
        #accuracy= tf.metrics.accuracy(labels=targets, predictions=prediction_label)

        correct_predictions = tf.equal(prediction_label,targets)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        if init_checkpoint:  # pretrained for bert
            tvars = tf.trainable_variables()
            (assignment_map,initialized_variable_names)=get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint,assignment_map)

        if mode == tf.estimator.ModeKeys.PREDICT:
            logging.info("**** Start predict ****")
            predictions= {"labels": prediction_label,'probabilities':probabilities}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)})

        elif mode == tf.estimator.ModeKeys.EVAL:
            logging.info("**** Start evaluate ****")
            eval_metric_ops = create_eval(targets, prediction_label)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

        else:
            logging.critical('**** Start training ****')
            #train_op=create_optimizer_basic_adam(loss,learning_rate=params['learning_rate'])
            logging_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": accuracy}, every_n_iter=100)
            num_train_steps = int(params['len_train_examples'] / params['batch_size'] * params['num_train_epochs'])
            train_op=create_optimizer_warmup_adam(loss=loss,
                                                  init_learning_rate=params['learning_rate'],
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=int(0.10*num_train_steps))
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[logging_hook])
    return model_fn
