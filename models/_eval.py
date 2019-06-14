import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,roc_auc_score
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


def metric_fn():
    pass


def create_eval(labels,predictions):
    accuracy=tf.metrics.accuracy(labels,predictions,name='acc_op')
    eval_metrics = {'acc': accuracy}
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
    #micro average is the same for p,r and f1, so use weighted f1 here
    accuracy = accuracy_score(labels,predictions)
    pre=precision_score(y_true=labels,y_pred=predictions,average='micro')
    rec=recall_score(y_true=labels,y_pred=predictions,average='micro')
    weighted_f1=f1_score(y_true=labels,y_pred=predictions,average='weighted')
    eval_metrics = {'accuracy': accuracy,'precision':pre,'recall':rec,'weighted_f1':weighted_f1}
    return eval_metrics


def get_metrics_ops(labels, predictions, num_labels):
    cm, op = _streaming_confusion_matrix(labels, predictions, num_labels)
    tf.logging.info(type(cm))
    tf.logging.info(type(op))
    return (tf.convert_to_tensor(cm), op)


def get_metrics(conf_mat, num_labels):
    precisions = []
    recalls = []
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * pre * rec / (pre + rec)
    return pre, rec, f1
