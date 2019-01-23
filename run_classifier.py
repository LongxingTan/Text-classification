
from models.cnn import *
from prepare_inputs import *
from models._optimization import *
import logging

# without logging configuration, tensorflow won't print the training information
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def prepare_input():
    online=OnlineProcessor(seq_lenth=params["seq_length"])
    train=online.get_train_examples(data_dir=params['data_dir'])
    dev=online.get_dev_axamples(data_dir=params['data_dir'])
    test=online.get_test_examples(data_dir=params['data_dir'])


def model_fn_builder(params):
    def model_fn(features, labels, mode):
        inputs, targets = features['input_ids'], features["label_ids"]
        model = TextCNN(params, mode == tf.estimator.ModeKeys.TRAIN)
        logits = model(inputs, targets)
        probabilities=tf.nn.softmax(logits,name='softmax_tensor')
        labels_pred = tf.argmax(logits, -1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"labels": labels_pred,
                           'probabilities':probabilities}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        targets_onehot = tf.one_hot(targets, depth=params['n_class'], dtype=tf.float32)
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets_onehot)
        loss = tf.reduce_mean(losses)
        tf.logging.info('loss {}'.format(loss))

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels=targets,
                                           predictions=labels_pred,
                                           name='acc_op')
            eval_metrics = {'accuracy': accuracy}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              predictions={"predictions": labels_pred},
                                              eval_metric_ops=eval_metrics)
        else:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    return model_fn

def run_classifier():
    cnn_model_fn=model_fn_builder(params)
    estimator= tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=params["model_dir"])
    #tensors_to_log={'probabilites':'softmax_tensor'}
    #logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)

    train_input_fn=file_based_input_fn_builder(input_file=os.path.join(params["data_dir"],"train.tf_record"),
                                               params=params,
                                               is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=1000)

    eval_input_fn=file_based_input_fn_builder(input_file=os.path.join(params['data_dir'],'eval.tf_record'),
                                              params=params,
                                              is_training=False)
    estimator.evaluate(input_fn=eval_input_fn)

    predict_input_fn=file_based_input_fn_builder(input_file=os.path.join(params['data_dir'],'test.tf_record'),
                                                 params=params,
                                                 is_training=False)
    estimator.predict(input_fn=predict_input_fn)


if __name__=="__main__":
    prepare_input()
    run_classifier()

