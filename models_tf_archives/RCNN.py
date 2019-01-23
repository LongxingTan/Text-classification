
import tensorflow as tf
import datetime
from models_tf_archives._utils import *


class RCNN():
    def __init__(self,session,x_train,y_train, x_test, y_test,vocabulary_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.sentence_length = config.sentence_length
        self.n_classes = len(np.unique(y_train))
        self.embedding_type = config.embedding_type
        self.embedding_dim = config.embedding_dim
        self.embedding_matrix = config.embedding_matrix
        self.vocabulary_size = vocabulary_size
        self.lstm_hidden_size=config.lstm_hidden_size
        self.dense_hidden_size=config.dense_hidden_size

        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.sess = session
        self.save_model='./rcnn.ckpt'

    def build(self):
        self.input_x=tf.placeholder(name='input_x',shape=(None,sentence_length),dtype=tf.int32)
        self.input_y=tf.placeholder(name='input_y',shape=(None,n_classes),dtype=tf.int32)
        self.keep_dropout_prob=tf.placeholder(name='dropout_keep_prob',dtype=tf.float32)
        self.global_step = tf.Variable(0, name='global_step')

        with tf.variable_scope('embed'):
            if self.embedding_matrix is False:
                W_embed=tf.Variable(name='embed_w',initial_value=tf.random_uniform(
                    [self.vocabulary_size,self.embedding_dim],-1.0,1.0),dtype=tf.float32)
            else:
                embedding_matrix = word_embed(word2index, self.embedding_type)
                W_embed=tf.get_variable(shape=[self.vocabulary_size,self.embedding_dim],name='embed_w',
                                        initializer=tf.constant_initializer(embedding_matrix),trainable=True)
            self.embed=tf.nn.embedding_lookup(W_embed,self.input_x) #[batch_size,sentence_length,embedding_dim]
            #print(self.embed.shape)

        with tf.variable_scope('bi-rnn'):
            fw_cell=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            fw_cell=tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=self.keep_dropout_prob)
            bw_cell=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            bw_cell=tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=self.keep_dropout_prob)
            (outputs_fw,outputs_bw),_=tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                      cell_bw=bw_cell,
                                                                      inputs=self.embed,
                                                                      dtype=tf.float32)
            #print(outputs_fw.shape)

        with tf.name_scope("context"):
            shape = [tf.shape(outputs_fw)[0], 1, tf.shape(outputs_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape), outputs_fw[:, :-1]], axis=1, name="context_left")
            self.c_right = tf.concat([outputs_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")
            #print(self.c_left.shape)

        with tf.name_scope("word-representation"):
            self.x = tf.concat([self.c_left, self.embed, self.c_right], axis=2, name="x")
            embedding_size = 2 * self.lstm_hidden_size + self.embedding_dim

        with tf.name_scope("text-representation"):
            W2 = tf.Variable(tf.random_uniform([embedding_size, self.dense_hidden_size], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.dense_hidden_size]), name="b2")
            self.y2 = tf.einsum('aij,jk->aik', self.x, W2) + b2

        with tf.name_scope("max-pooling"):
            self.y3 = tf.reduce_max(self.y2, axis=1)

        with tf.name_scope("output"):
            W4 = tf.get_variable("W4", shape=[self.dense_hidden_size, n_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b4")
            self.logits = tf.nn.xw_plus_b(self.y3, W4, b4, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.softmax = tf.nn.softmax(self.logits, name='softmax')
            self.predict_prob = tf.reduce_max(self.softmax, 1, name='predict_prob')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                          global_step=self.global_step)
        self.saver = tf.train.Saver()

    def train(self, mode=None, restore=False):
        if mode != 'continue':
            logging.info('Model building ...')
            self.build()
            self.sess.run(tf.global_variables_initializer())
        else:
            if restore:
                self.build()
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(sess=self.sess, save_path=self.save_model)

        logging.info('Start training...')
        for epoch in range(self.n_epochs):
            loss_epoch = []
            acc_epoch = []
            batches = create_batch(list(zip(self.x_train, self.y_train)), batch_size=self.batch_size,n_epochs=1)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.keep_dropout_prob: 0.5}
                _, loss, step, acc = self.sess.run([self.train_op, self.loss, self.global_step, self.accuracy, ],
                                              feed_dict=feed_dict)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                print("Epoch %d, Step: %d, Loss: %.4f, Acc: %.4f\r" % (epoch, step, loss, acc))
            print("Epoch %d, Step: %d, Loss: %.4f, Acc: %.4f\r" % (epoch, step, np.mean(loss_epoch), np.mean(acc_epoch)))

    def test(self, sub_size=None, restore=False):
        if sub_size == None:
            test_sub_size = len(self.y_test)
        else:
            test_sub_size = sub_size

        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess, save_path=self.save_model)

        loss_epoch = []
        acc_epoch = []
        batches = create_batch(list(zip(self.x_test, self.y_test)), batch_size=test_sub_size, n_epochs=1)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.keep_dropout_prob: 1}
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
        print("Loss: %.4f, Acc: %.4f\r" % (np.mean(loss_epoch), np.mean(acc_epoch)))


    def predict(self, x_new, restore=False):
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess, save_path=self.save_model)

        feed_dict = {self.input_x: x_new, self.keep_dropout_prob: 1.0}
        pred, prob = self.sess.run([self.predictions, self.predict_prob], feed_dict=feed_dict)
        return pred, prob

if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('../CAC.csv')
    #x_data, y_data = x_data[:100], y_data[:100] ##delete
    #n_classes = len(np.unique(y_data)) ##delete
    x_token = x_data.apply(tokenize_sentence)
    x_token_pad, sentence_length = pad_sentence(x_token,sentence_length=config.sentence_length)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    # embedding_matrix = word_embedding(word2index)
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        rcnn=RCNN(sess,x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,vocabulary_size=len(word2index))
        rcnn.train()
        rcnn.test()

        y_new_index, y_new_prob = rcnn.predict(x_test)
        sess.close()

        y_new_label = [index2label[i] for i in y_new_index]
        x_new_label = [[index2word[i] for i in sentence] for sentence in x_test]
        y_test_index = np.argmax(y_test, 1)
        y_test_label = [index2label[i] for i in y_test_index]
        with open('rcnn_result.csv', 'w', newline='', encoding='utf-8-sig') as file:
            w = csv.writer(file, delimiter=';')
            w.writerow(('Words', 'Label', 'Label_predicted', 'Probability'))
            w.writerows(zip(x_new_label, y_test_label, y_new_label, y_new_prob))


'''
def RCNN_train(x_train,y_train,x_test,y_test,sentence_length,vocabulary_size, embedding_dim, hidden_size,lstm_size, n_classes,embedding_matrix,batch_size,n_epochs):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn=RCNN(sentence_length=sentence_length,vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
                     lstm_size=lstm_size,hidden_size=hidden_size, n_classes=n_classes,embedding_matrix=embedding_matrix)
            global_step=tf.Variable(0,name='global_step',trainable=False)
            train_op=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(rnn.loss,global_step=global_step)

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            batches=create_batch(list(zip(x_train,y_train)), batch_size=batch_size, n_epochs=n_epochs)
            for batch in batches:
                x_batch,y_batch=zip(*batch)
                feed_dict={rnn.input_x:x_batch,
                           rnn.input_y:y_batch,
                           rnn.dropout_keep_prob:0.5}
                _,step,loss,accuracy=sess.run([train_op,global_step,rnn.loss,rnn.accuracy],feed_dict=feed_dict)


                if step%10==0:
                    saver.save(sess, './tf_lstmnet.ckpt')
                    time_str=datetime.datetime.now().isoformat()
                    feed_dict_test={rnn.input_x:x_test,
                                    rnn.input_y:y_test,
                                    rnn.dropout_keep_prob:1.0}
                    test_accuracy=sess.run([rnn.accuracy],feed_dict=feed_dict_test)
                    logging.critical("{}: step {}, loss {}, acc {},test_acc {}".format(time_str, step, loss, accuracy,test_accuracy))

def RCNN_predict_new_data(x_new):
    pass

if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('../CAC.csv')
    #x_data, y_data = x_data[:100], y_data[:100] ##delete
    #n_classes = len(np.unique(y_data)) ##delete
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

    RCNN_train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,sentence_length=sentence_length,
               vocabulary_size=len(word2index),embedding_dim=300, lstm_size=128,hidden_size=128,
               n_classes=n_classes, embedding_matrix=None,batch_size=64,n_epochs=20)

#code reference:
#https://github.com/roomylee/rcnn-text-classification/blob/master/rcnn.py
'''