import tensorflow as tf


def create_loss(logits,y_onehot,loss_type):
    if loss_type == 'cross_entropy':
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_onehot)
        loss = tf.reduce_mean(losses)
        return loss

    elif loss_type=='weighted_cross_entropy':
        y_pred=tf.nn.softmax(logits)
        losses=tf.nn.weighted_cross_entropy_with_logits(targets=y_onehot,logits=y_pred)
        loss=tf.reduce_mean(losses)
        return loss

    elif loss_type == 'margin_loss':
        y_pred = tf.nn.softmax(logits)
        y_true = tf.cast(y_onehot, tf.float32)
        losses = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.25 * (1 - y_true) * tf.square(
            tf.maximum(0., y_pred - 0.1))
        loss = tf.reduce_mean(losses)
        return loss

    elif loss_type=='contrastive_loss':
        margin=1
        y_pred = tf.nn.softmax(logits)
        y_true=tf.cast(y_onehot, tf.float32)
        losses=y_true*tf.square(y_pred)+(1-y_true)*tf.square(tf.maximum(margin-y_pred,0))
        loss=tf.reduce_mean(losses)
        return loss

    elif loss_type=='smape_loss':
        #smape_loss(true, predicted, weights):
        """
        Differentiable SMAPE loss
        :param true: Truth values
        :param predicted: Predicted values
        :param weights: Weights mask to exclude some values
        :return:
        """
        epsilon = 0.1  # Smoothing factor, helps SMAPE to be well-behaved near zero
        y_pred = tf.nn.softmax(logits)
        y_true = tf.cast(y_onehot, tf.float32)
        true_o = tf.expm1(y_true)
        pred_o = tf.expm1(y_pred)
        summ = tf.maximum(tf.abs(true_o) + tf.abs(pred_o) + epsilon, 0.5 + epsilon)
        smape = tf.abs(pred_o - true_o) / summ * 2.0
        return tf.losses.compute_weighted_loss(smape, weights=None, loss_collection=None)

    elif loss_type=='multi_task_loss':
        loss=[]
        for i in range(len(y_onehot)):
            with tf.name_scope("loss" + str(i)):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits[i], y_onehot[i])
                loss.append(tf.reduce_mean(losses))
        return loss
