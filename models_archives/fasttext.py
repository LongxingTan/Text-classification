import tensorflow as tf
import numpy as np
from models_archives._utils import *

class fasttext():
    def __init__(self,sentence_length,vocabulary_size,embedding_dim,n_classes,embedding_matrix=None):
        self.input_x=tf.placeholder(tf.int32,[None, sentence_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None, n_classes],name='input_y')

        with tf.name_scope('embedding'):
            if embedding_matrix is None:
                self.EmbeddingW=tf.get_variable('EmbeddingW',[vocabulary_size,embedding_dim])
            else:
                pass
            self.W=tf.get_variable('W',[embedding_dim,n_classes])
            self.b=tf.get_variable('b',[n_classes])

        self.embedding_chars=tf.nn.embedding_lookup(self.EmbeddingW,self.input_x)
        self.embedding_chars_mean=tf.reduce_mean(self.embedding_chars,axis=1)

        logits=tf.matmul(self.embedding_chars_mean,self.W)+self.b
        predictions = tf.argmax(logits, axis=1, name="predictions")

        with tf.name_scope('loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=logits)
            self.loss = tf.reduce_sum(loss, axis=1)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(predictions, tf.arg_max(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

def fasttext_train(x_train, y_train, x_test, y_test,sentence_lenth,vocabulary_size,embedding_dim,n_classes,embedding_matrix,batch_size):
    with tf.Session() as sess:
        fasttext_model=fasttext(sentence_length=sentence_lenth,vocabulary_size=vocabulary_size,embedding_dim=embedding_dim,
                                n_classes=n_classes,embedding_matrix=None)
        global_step = tf.Variable(0, name='global_stop', trainable=False)
        train_op = tf.train.AdamOptimizer(10e-4).minimize(fasttext_model.loss)

        sess.run(tf.global_variables_initializer())
        batches = create_batch(list(zip(x_train, y_train)), batch_size=batch_size, n_epochs=25)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {fasttext_model.input_x: x_batch,
                         fasttext_model.input_y: y_batch}
            _, step, loss, accuracy = sess.run([train_op, global_step, fasttext_model.loss, fasttext_model.accuracy], feed_dict)
            print("step:",step,"loss:",loss,"accuracy",accuracy)

def fasttext_predict_new_data(x_new):
    pass

if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('CAC.csv')
    x_data, y_data = x_data[:100], y_data[:100]
    x_token = x_data.apply(tokenize_sentence)
    sentence_padded, sentence_length=pad_sentence(x_token)
    x_token_pad, sentence_length = pad_sentence(x_token)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    #embedding_matrix = word_embedding(word2index)
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)
    print(np.array(x_train).shape,np.array(y_train).shape)

    fasttext_train(x_train, y_train, x_test, y_test,sentence_length, vocabulary_size=len(word2index), embedding_dim=300,
                  n_classes=n_classes,embedding_matrix=None, batch_size=64)

#code reference
#https://github.com/brightmart/text_classification/blob/master/a01_FastText/p5_fastTextB_model.py
#https://github.com/cccadet/Machine-Learning/blob/d8bd217c082e0ee5b0b49aa24bea5f0249843bdb/Feature%20Engineering%20Text%20Data%20-%20Advanced%20Deep%20Learning%20Strategies.ipynb