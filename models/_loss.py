import tensorflow as tf


def create_loss(logits,y_onehot,loss_type):
    if loss_type == 'spread_loss':
        pass

    elif loss_type == 'cross_entropy':
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot)
        loss = tf.reduce_mean(losses)

        return loss

    elif loss_type == 'margin_loss':
        y_labels = tf.cast(y_labels, tf.float32)
        losses = y_labels * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.25 * (1 - y_labels) * tf.square(
            tf.maximum(0., y_pred - 0.1))
        loss = tf.reduce_mean(losses)
        return loss
