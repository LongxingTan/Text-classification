import tensorflow as tf
from models_archives._utils import *

class Config():
    def __init__(self):
        self.embedding_dim=300
        self.embedding_matrix=False #False or true
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

class GRU_Attention():
    def __init__(self,session,x_train,y_train, x_test, y_test,vocabulary_size,tfidf_feature):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.sentence_length=config.sentence_length
        self.n_classes=len(np.unique(y_train))
        self.embedding_type=config.embedding_type
        self.embedding_dim=config.embedding_dim
        self.embedding_matrix=config.embedding_matrix
        self.vocabulary_size = vocabulary_size
        self.gru_hidden_size = config.gru_hidden_size
        self.attention_hidden_size=config.attention_hidden_size
        self.batch_size=config.batch_size
        self.n_epochs=config.n_epochs
        self.learning_rate=config.learning_rate
        self.tfidf_feature=tfidf_feature
        self.sess=session
        self.save_path = './gru_att.ckpt'


    def build(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.n_classes], name='input_y')
        self.keep_dropout_prob = tf.placeholder(tf.float32,name='dropout')
        self.global_step=tf.Variable(0,name='global_step')

        with tf.name_scope('embedding'):
            if self.embedding_matrix is False:
                self.embedding_W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dim], -1.0, 1.0),
                                               name='embedding_W')
            else:
                self.embedding_matrix=word_embed_trans(word2index,'word2vec')
                self.embedding_W=tf.get_variable(name='embedding_W',shape=[self.vocabulary_size,self.embedding_dim],initializer=tf.constant_initializer(self.embedding_matrix),trainable=True)

            self.embedding_out=tf.nn.embedding_lookup(self.embedding_W,self.input_x)

        with tf.name_scope('tfidf'):
            if config.tfidf==True:
                tfidf_matrix = use_tfidf_feature(word2index, self.tfidf_feature)
                tfidf_chars = tf.nn.embedding_lookup(tfidf_matrix, self.input_x)
                self.tfidf_chars_expanded = tf.cast(tfidf_chars,tf.float32)
                self.embedding_out = tf.concat([self.embedding_out, self.tfidf_chars_expanded], 2)

        self.encoder_input_embeded=tf.nn.dropout(self.embedding_out,keep_prob=1.0)

        encoder_fw=tf.nn.rnn_cell.GRUCell(self.gru_hidden_size)
        encoder_bw=tf.nn.rnn_cell.GRUCell(self.gru_hidden_size)
        self.encoder_fw=tf.nn.rnn_cell.DropoutWrapper(encoder_fw,input_keep_prob=self.keep_dropout_prob)
        self.encoder_bw=tf.nn.rnn_cell.DropoutWrapper(encoder_bw,input_keep_prob=self.keep_dropout_prob)

        with tf.variable_scope('bi_gru') as scope:
            gru_outputs,gru_states=tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw,cell_bw=self.encoder_bw,
                                                                   inputs=self.encoder_input_embeded,dtype=tf.float32)
            self.encoder_output=tf.concat(gru_outputs,2)
            self.encoder_state=tf.concat(gru_states,1)

        with tf.variable_scope('attention') as scope:
            self._atn_in=tf.expand_dims(self.encoder_output,axis=2)
            self.atn_w=tf.Variable(tf.truncated_normal(shape=[1,1,2*self.gru_hidden_size,self.attention_hidden_size],
                                                       stddev=0.1),name='atn_w')
            self.atn_b=tf.Variable(tf.zeros(shape=[self.attention_hidden_size]))
            self.atn_v=tf.Variable(tf.truncated_normal(shape=[1,1,self.attention_hidden_size,1],stddev=0.1),name='atn_v')
            self.atn_activation=tf.nn.tanh(tf.nn.conv2d(self._atn_in,self.atn_w,strides=[1,1,1,1],padding='SAME')+self.atn_b)
            self.atn_scores=tf.nn.conv2d(self.atn_activation,self.atn_v,strides=[1,1,1,1],padding='SAME')
            atn_probs=tf.nn.softmax(tf.squeeze(self.atn_scores,[2,3]))
            _atn_out=tf.matmul(tf.expand_dims(atn_probs,1),self.encoder_output)
            self.attention_output=tf.squeeze(_atn_out,[1],name='atn_out')

        with tf.variable_scope('output') as scope:
            self.output_w=tf.Variable(tf.truncated_normal(shape=(self.gru_hidden_size*2,self.n_classes),stddev=0.1),name='output_w')
            self.output_b=tf.Variable(tf.zeros(self.n_classes),name='output_b')
            self.logits=tf.matmul(self.attention_output,self.output_w)+self.output_b
            self.prediction=tf.cast(tf.argmax(self.logits,1),tf.int32)
            self.softmax = tf.nn.softmax(self.logits, name='softmax')
            self.predict_prob = tf.reduce_max(self.softmax, 1, name='predict_prob')

            input_label=tf.cast(tf.argmax(self.input_y, 1),tf.int32)
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,input_label),tf.float32))
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.input_y))

        self.train_op=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)
        self.saver=tf.train.Saver()



    def train(self,mode=None,restore=False):
        if mode!='continue':
            logging.info('Model building ...')
            self.build()
            self.sess.run(tf.global_variables_initializer())
        else:
            if restore:
                self.build()
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(sess=self.sess,save_path=self.save_path)

        logging.info('Start training...')
        for epoch in range(self.n_epochs):
            loss_epoch=[]
            acc_epoch=[]
            batches = create_batch(list(zip(self.x_train, self.y_train)), batch_size=self.batch_size, n_epochs=1)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict={self.input_x:x_batch,self.input_y:y_batch,self.keep_dropout_prob:0.5}
                _,loss,step,acc=sess.run([self.train_op,self.loss,self.global_step,self.accuracy,],feed_dict=feed_dict)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                print("Epoch %d, Step: %d, Loss: %.4f, Acc: %.4f\r" % (epoch, step, loss, acc))
            logging.info("Epoch %d, Step: %d, Loss: %.4f, Acc: %.4f\r" % (epoch, step, np.mean(loss_epoch), np.mean(acc_epoch)))

    def test(self, sub_size=None,restore=False):
        if sub_size==None:
            test_sub_size=len(self.y_test)
        else:
            test_sub_size=sub_size

        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess,save_path=self.save_path)

        loss_epoch = []
        acc_epoch = []
        batches = create_batch(list(zip(self.x_test, self.y_test)), batch_size=test_sub_size, n_epochs=1)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.keep_dropout_prob: 1}
            loss, acc = sess.run([ self.loss, self.accuracy ], feed_dict=feed_dict)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
        logging.info("Loss: %.4f, Acc: %.4f\r" % (np.mean(loss_epoch), np.mean(acc_epoch)))

    def predict(self,x_new, restore=False):
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess, save_path=self.save_path)

        feed_dict={self.input_x:x_new,self.keep_dropout_prob:1.0}
        pred,prob=self.sess.run([self.prediction,self.predict_prob],feed_dict=feed_dict)
        return pred,prob


if '__main__' == __name__:
    x_train, x_test, y_train, y_test, word2index, n_classes, index2word, index2label=create_data(sentence_length=config.sentence_length,sample=None)
    sess = tf.Session()
    gru_att=GRU_Attention(sess,x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,vocabulary_size=len(word2index),tfidf_feature=tfidf_feature)
    gru_att.train()
    gru_att.test()
    y_new_index,y_new_prob=gru_att.predict(x_test)
    sess.close()

    y_new_label = [index2label[i] for i in y_new_index]
    x_new_label = [[index2word[i] for i in sentence] for sentence in x_test]
    y_test_index = np.argmax(y_test, 1)
    y_test_label = [index2label[i] for i in y_test_index]
    with open('gru_att_result.csv', 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file, delimiter=';')
        w.writerow(( 'Words', 'Label', 'Label_predicted', 'Probability'))
        w.writerows(zip(x_new_label, y_test_label, y_new_label, y_new_prob))