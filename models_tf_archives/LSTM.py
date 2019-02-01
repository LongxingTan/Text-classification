import tensorflow as tf
from models_tf_archives._utils import *
import datetime
import os


class Config():
    def __init__(self):
        self.embedding_dim=300
        self.embedding_matrix=False   #False or true
        self.embedding_type='word2vec'
        self.sentence_length=50
        self.tfidf=True
        self.chi_k=400

        self.lstm_hidden_size=300
        self.gru_hidden_size=128
        self.attention_hidden_size=64
        self.l2_reg=0.0

        self.learning_rate=10e-3
        self.batch_size=128
        self.n_epochs=10
        self.save_per_epoch=10

        self.filter_sizes=[2,4,6]
        self.filter_num=16

config = Config()

class biLSTM():
    def __init__(self,sess,x_train,x_test,y_train,y_test,vocabulary_size,tfidf_feature):
        self.sess=sess
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.sentence_length=config.sentence_length
        self.n_classes=len(np.unique(y_train))
        self.vocabulary_size=vocabulary_size
        self.embedding_dim=config.embedding_dim
        self.embedding_matrix=config.embedding_matrix
        self.lstm_hidden_size=config.lstm_hidden_size
        self.l2_reg=config.l2_reg
        self.n_epochs=config.n_epochs
        self.batch_size=config.batch_size
        self.tfidf_feature=tfidf_feature
        self.save_path= './lstmnet.ckpt'


    def build(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.n_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        l2_loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            if self.embedding_matrix is False:
                self.W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dim], -1.0, 1.0), name='embedding_W')
            else:
                embedding_matrix = word_embed_trans(word2index, 'word2vec')
                self.W = tf.get_variable(name="W", shape=[self.vocabulary_size, self.embedding_dim],
                                         initializer=tf.constant_initializer(embedding_matrix), trainable=True)
            self.embedding_chars=tf.nn.embedding_lookup(self.W,self.input_x)

        with tf.name_scope('tfidf'):
            if config.tfidf==True:
                tfidf_matrix = use_tfidf_feature(word2index, self.tfidf_feature)
                tfidf_chars = tf.nn.embedding_lookup(tfidf_matrix, self.input_x)
                self.tfidf_chars_expanded = tf.cast(tfidf_chars,tf.float32)
                self.embedding_chars = tf.concat([self.embedding_chars, self.tfidf_chars_expanded], 2)


        with tf.name_scope('bi_lstm'):
            cell_fw=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
            cell_bw=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            cell_bw=tf.nn.rnn_cell.DropoutWrapper(cell_bw,output_keep_prob=self.dropout_keep_prob)
            all_outputs,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=self.embedding_chars,
                                            sequence_length=None,dtype=tf.float32)
            all_outputs=tf.concat(all_outputs,2)
            self.h_outputs=all_outputs[:,-1,:]
            #self.h_outputs=self.last_relevant(all_outputs,sentence_length)

        with tf.name_scope('output'):
            W_out = tf.get_variable("W_out", shape=[2*self.lstm_hidden_size, self.n_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W_out)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W_out, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            softmax=tf.nn.softmax(self.logits,name='softmax')
            self.predict_prob=tf.reduce_max(softmax,1,name='predict_prob')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_loss*self.l2_reg

        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,axis=1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,tf.float32),name='accuracy')

        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver()


    def train(self,mode=None,restore=False):
        if mode!='continue':
            logging.info('Model building ....')
            self.build()
            self.sess.run(tf.global_variables_initializer())
        else:
            if restore:
                self.build()
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(sess=self.sess,save_path=self.save_path)

        logging.info('Start training ...')
        for epoch in range(self.n_epochs):
            loss_epoch,acc_epoch=[],[]
            batches = create_batch(list(zip(x_train, y_train)), batch_size=self.batch_size, n_epochs=1)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {self.input_x: x_batch,
                             self.input_y: y_batch,
                             self.dropout_keep_prob: 0.5}
                _, step, loss, accuracy = self.sess.run([self.train_op, self.global_step, self.loss,self.accuracy], feed_dict=feed_dict)
                loss_epoch.append(loss)
                acc_epoch.append(accuracy)
                if step % 10 == 0:
                    print("Epoch {},step {},loss {}, acc {}".format(epoch,step, loss, accuracy))
            logging.info("Epoch {},step {},epoch_loss {}, epoch_acc {}".format(epoch,step, np.mean(loss_epoch), np.mean(acc_epoch)))

            best_acc=0
            test_accuracy=self.test()
            if test_accuracy>best_acc:
                self.saver.save(self.sess,self.save_path)
                best_acc=test_accuracy
                print('Test accuracy',best_acc)


    def test(self,sub_size=None,restore=False):
        if sub_size==None:
            sub_size=len(self.y_test)
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer)
            self.saver.restore(sess=self.sess,save_path=self.save_path)

        loss_epoch,acc_epoch=[],[]
        batches=create_batch(list(zip(self.x_test,self.y_test)),batch_size=sub_size,n_epochs=1)
        for batch in batches:
            x_batch,y_batch=zip(*batch)
            feed_dict={self.input_x:x_batch,self.input_y:y_batch,self.dropout_keep_prob:1.0}
            loss,acc=self.sess.run([self.loss,self.accuracy],feed_dict=feed_dict)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
        logging.info('Loss: %.4f, Acc: %.4f\r'%(np.mean(loss_epoch),np.mean(acc_epoch)))
        return np.mean(acc_epoch)


    def predict(self,x_new,restore=False):
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer)
            self.saver.restore(sess=self.sess,save_path=self.save_path)
        feed_dict={self.input_x:x_new,self.dropout_keep_prob:1.0}
        pred,prob=self.sess.run([self.predictions,self.predict_prob],feed_dict=feed_dict)
        return pred,prob

    def last_relevant(self,seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)


if '__main__' == __name__:
    x_train, x_test, y_train, y_test, word2index,n_classes,index2word,index2label=create_data(sentence_length=config.sentence_length,sample=None)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    configgpu = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=configgpu)

    bilstm=biLSTM(sess,x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,vocabulary_size=len(word2index),tfidf_feature=False)
    bilstm.train()
    bilstm.test()
    y_new_index,y_new_prob=bilstm.predict(x_test)
    sess.close()

    y_new_label = [index2label[i] for i in y_new_index]
    x_new_label = [[index2word[i] for i in sentence] for sentence in x_test]
    y_test_index = np.argmax(y_test, 1)
    y_test_label = [index2label[i] for i in y_test_index]
    with open('biLSTM_results.csv', 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file, delimiter=';')
        w.writerow(( 'Words', 'Label', 'Label_predicted', 'Probability'))
        w.writerows(zip(x_new_label, y_test_label, y_new_label, y_new_prob))

