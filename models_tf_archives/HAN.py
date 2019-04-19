#paper reference:
##http://www.aclweb.org/anthology/N16-1174
#code reference:
#https://github.com/indiejoseph/doc-han-att/blob/master/model.py

import tensorflow as tf
from models_tf_archives._utils import *


class Config():
    def __init__(self):
        self.embedding_dim=300
        self.embedding_matrix=False   #False or true
        self.embedding_type = 'word2vec'
        self.sentence_length=50
        self.vocabulary_size=None
        self.filter_sizes=[2,4,6]
        self.filter_num=16
        self.learning_rate=10e-3
        self.batch_size=64
        self.n_epochs=15
        self.attention_dim=128
        self.lstm_dim=128
        self.save_path='./HANnet.ckpt'
        self.penalization=True
config = Config()


class HAN(object):
    def __init__(self,session,x_train,y_train,x_test,y_test,n_sentences, each_sentence_length,vocabulary_size):
        self.sess=session
        self.x_train=x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_sentences=n_sentences
        self.each_sentence_length=each_sentence_length
        self.n_classes=len(np.unique(y_train))
        self.embedding_type = config.embedding_type
        self.embedding_dim = config.embedding_dim
        self.embedding_matrix=config.embedding_matrix
        self.lstm_dim=config.lstm_dim
        self.attention_dim=config.attention_dim
        self.vocabulary_size = vocabulary_size

        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.save_path='./HANnet.ckpt'


    def build(self):
        self.input_x=tf.placeholder(shape=(None,n_sentences,each_sentence_length),dtype=tf.int32,name='input_x')
        self.input_y=tf.placeholder(shape=(None,n_classes),dtype=tf.int32,name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='drop_out')
        self.global_step = tf.Variable(0, name='global_step')

        with tf.name_scope('embedding'):
            if self.embedding_matrix is False:
                self.embed_W=tf.Variable(tf.random_uniform([self.vocabulary_size,self.embedding_dim],-1.0,1.0),name='embed_W')
            else:
                embedding_matrix = word_embed_trans(word2index, self.embedding_type)
                self.embed_W=tf.get_variable(shape=[self.vocabulary_size,self.embedding_dim],
                                             initializer=tf.constant_initializer(embedding_matrix),
                                             trainable=True,name='embed_W')
            embedding=tf.nn.embedding_lookup(self.embed_W,self.input_x)

        word_inputs=tf.reshape(embedding,[-1, each_sentence_length, self.embedding_dim])
        with tf.variable_scope('word_encoder') as scope:
            word_outputs,_=self.bi_gru_encode(word_inputs,self.lstm_dim,scope)

            with tf.name_scope('attention'):
                word_attention = self.attention(word_outputs,attention_dim=self.attention_dim)
                word_attention=tf.reshape(word_attention,[-1,n_sentences,2*self.lstm_dim])

            with tf.name_scope('dropout'):
                word_attention_drop =tf.nn.dropout(word_attention,self.dropout_keep_prob)

        with tf.variable_scope ('sentence_encoder') as scope:
            sentence_outputs,_=self.bi_gru_encode(word_attention_drop,self.lstm_dim,scope)
            sentence_outputs.set_shape([None, n_sentences, 2 * self.lstm_dim])

            with tf.name_scope('attention'):
                sentence_attention=self.attention(sentence_outputs,attention_dim=self.attention_dim)

            with tf.name_scope('dropout'):
                sentence_attention_drop=tf.nn.dropout(sentence_attention,self.dropout_keep_prob)

        with tf.name_scope('output'):
            W=tf.get_variable('W',shape=[2*self.lstm_dim,n_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[n_classes]),name='b')
            self.logits=tf.nn.xw_plus_b(sentence_attention_drop,W,b,name='scores')
            self.prediction=tf.argmax(self.logits,1,name='predictions')
            softmax = tf.nn.softmax(self.logits, name='softmax')
            self.predict_prob = tf.reduce_max(softmax, 1, name='predict_prob')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')

        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver()



    def train(self,mode=None,restore=False):
        if mode!='continue':
            logging.info('Model building ....')
            self.build()
            self.sess.run(tf.global_variables_initializer())
        else:
            if restore:
                logging.info('Model restoring ....')
                self.build()
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(sess=self.sess,save_path='./HANnet.ckpt')

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
                self.saver.save(self.sess, self.save_path)
                best_acc=test_accuracy
                logging.info('Test accuracy {}',best_acc)

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
            self.saver.restore(sess=self.sess,save_path='./HANnet.ckpt')
        feed_dict={self.input_x:x_new,self.dropout_keep_prob:1.0}
        pred,prob=self.sess.run([self.prediction,self.predict_prob],feed_dict=feed_dict)
        return pred,prob


    def bi_gru_encode(self, inputs, lstm_size,scope=None):
        with tf.variable_scope(scope or 'bi_gru'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
            cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_size)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

            enc_out,(enc_state_fw,enc_state_bw)=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,
                                                                                inputs=inputs,sequence_length=None,
                                                                                dtype=tf.float32,scope='L')
            enc_state=tf.concat([enc_state_bw,enc_state_bw],1)
            enc_outputs=tf.concat(enc_out,2)
        return enc_outputs,enc_state

    def attention(self,inputs,attention_dim):
        with tf.name_scope('attention__'):
            attention_vector=tf.get_variable(shape=[attention_dim],dtype=tf.float32,name='attention_vector')
            input_projection=tf.contrib.layers.fully_connected(inputs,attention_dim,activation_fn=tf.tanh)
            vector_att=tf.reduce_sum(tf.multiply(input_projection,attention_vector),axis=2,keep_dims=True)
            attention_weights=tf.nn.softmax(vector_att,dim=1)
            weighted_projection=tf.multiply(inputs,attention_weights)
            attention_outputs=tf.reduce_sum(weighted_projection,axis=1)
        return attention_outputs



if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('../CAC.csv')
    x_sent = x_data.apply(tokenize_text)
    x_token = [[list(map(tokenize_sentence, sentence)) for sentence in text] for text in x_sent]
    x_token_han=[[sentence[0].split(' ') for sentence in text] for text in x_token]
    word2index, index2word = word_index_transform([sentence[0].split(' ') for text in x_token for sentence in text])

    n_sentences=5
    each_sentence_length=15
    x_token_index=np.zeros((len(x_token_han),n_sentences,each_sentence_length),dtype='int32')
    for i,text in enumerate(x_token_han):
        for j,sentence in enumerate(text):
            if j<n_sentences:
                for k,word in enumerate(sentence):
                    if k<each_sentence_length:
                        x_token_index[i,j,k]=word2index[word]

    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)

    if tf.test.is_gpu_available:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        configgpu = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options,log_device_placement=False)
        sess = tf.Session(config=configgpu)
    else:
        sess=tf.Session()

    bilstm = HAN(sess, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                    vocabulary_size=len(word2index),n_sentences=n_sentences, each_sentence_length=each_sentence_length)
    bilstm.train()
    bilstm.test()
    y_new_index, y_new_prob = bilstm.predict(x_test)
    sess.close()

    y_new_label = [index2label[i] for i in y_new_index]
    '''
    x_new_label = [[index2word[i] for i in sentence] for sentence in x_test]
    y_test_index = np.argmax(y_test, 1)
    y_test_label = [index2label[i] for i in y_test_index]
    with open('HAN_results.csv', 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file, delimiter=';')
        w.writerow(('Words', 'Label', 'Label_predicted', 'Probability'))
        w.writerows(zip(x_new_label, y_test_label, y_new_label, y_new_prob))
'''