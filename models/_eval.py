import tensorflow as tf
from models._model_params import params
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score

def create_eval(lables,predictions):
    accuracy=tf.metrics.accuracy(lables,predictions)
    eval_metrics = {'accuracy': accuracy}
    return eval_metrics





def create_eval_binary(labels,predictions):
    labels=tf.cast(labels,tf.int16)
    predictions=tf.cast(predictions,tf.int16)

    accuracy = tf.metrics.accuracy(labels=labels,
                                                  predictions=predictions,
                                                  name='acc_op')
    micro_f1=tf.contrib.metrics.f1_score(labels=labels,
                                         predictions=predictions,
                                         name='f1')
    auc=tf.metrics.auc(labels=labels,
                       predictions=predictions,
                       name='auc')

    tf.summary.scalar('accuracy', accuracy[1])
    eval_metrics = {'accuracy': accuracy,'f1':micro_f1,'auc':auc}

    return eval_metrics

def create_eval_sk(labels,predictions):

    accuracy = accuracy_score(labels,predictions)
    micro_f1=f1_score(labels,predictions,average='micro')
    auc=roc_auc_score(labels,predictions,average='micro')
    eval_metrics = {'accuracy': accuracy,'f1':micro_f1,'auc':auc}

    return eval_metrics