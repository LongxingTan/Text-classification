#Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
#https://arxiv.org/pdf/1703.03130.pdf

import tensorflow as tf
from models_tf_archives._utils import *



class Config():
    def __init__(self):
        self.embedding_dim=300
        self.embedding_matrix=False   #False or true
        self.embedding_type = 'word2vec'
        self.sentence_length=50
        self.vocab_size=None
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

class Self_attention:
    def __init__(self,sess):
        self.session=sess
        self.sentence_length=config.sentence_length
        self.vocab_size=config.vocab_size
        self.embedding_size=config.embedding_dim
        self.lstm_hidden=config.lstm_dim
        self.att_dim=config.attention_dim
        self.n_classes=config.classes
        self.learning_rate=config.learning_rate
        self.n_epochs=config.n_epochs
        self.batch_size=config.batch_size
        self.penalization=config.penalization
        self.save_path='./self_att.ckpt'


    def build(self):
        self.input_x=tf.placeholder(tf.int32,[None,self.sentence_length],name='input_x')
        self.input_y=tf.placeholder(tf.int32,[None,self.n_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep')
        self.global_step=tf.Variable(0,name='global_step')

        with tf.variable_scope('embed'):
            if config.embedding_type=='random word2vec':
                self.embed_w=tf.Variable(tf.random_uniform((self.vocab_size,self.embedding_size),-1.0,1.0),name='embed_w')

            embed_encoder=tf.nn.embedding_lookup(self.embed_w,self.input_x) #shape [Batch_size,n_sentences, embedding_dim]

        with tf.variable_scope('lstm'):
            fw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden)
            bw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden)
            fw_cell_drop=tf.nn.rnn_cell.DropoutWrapper(fw_cell,input_keep_prob=self.dropout_keep_prob)
            bw_cell_drop=tf.nn.rnn_cell.DropoutWrapper(bw_cell,input_keep_prob=self.dropout_keep_prob)
            lstm_out,_=tf.nn.bidirectional_dynamic_rnn(fw_cell_drop,bw_cell_drop,inputs=embed_encoder,dtype=tf.float32)
            lstm_encoder=tf.concat([lstm_out[0],lstm_out[1]],axis=2)  #shape [Batch_size,n_sentences,2*lstm_hidden_size]

        with tf.variable_scope('attention'):
            attention_w1=tf.Variable(tf.random_normal((2*self.lstm_hidden,self.att_dim),stddev=0.1))
            attention_b1=tf.Variable(tf.random_normal([self.att_dim],stddev=0.1))

            att=tf.tanh(tf.add(tf.tensordot(lstm_encoder,attention_w1,axes=1),attention_b1)) #shape [batch_size,n_sentences,att_size]
            attention_w2=tf.Variable(tf.random_normal([self.att_dim],stddev=0.1))

            att_alpha_pre=tf.nn.softmax(tf.tensordot(att,attention_w2,axes=1),name='alpha')  #shape [batch_size,n_sentences]
            att_alpha=tf.expand_dims(att_alpha_pre,-1)
            att_output=tf.reduce_sum(lstm_encoder*att_alpha,1) #shape [batch_size,2*lstm_hidden]

        with tf.variable_scope('penalization'):
            if self.penalization:
                tile_eye=tf.reshape(tf.tile(tf.eye(self.sentence_length),[self.batch_size,1]),[-1,self.sentence_length,self.sentence_length])
                Attention_t=tf.matmul(tf.transpose(att_output,[0,2,1]),att_output)-tile_eye
                self.penalize=tf.square(tf.norm(Attention_t,axis=[-2,-1],ord='fro'))


        with tf.variable_scope('output'):
            output_w=tf.Variable(tf.truncated_normal(shape=(self.lstm_hidden*2,self.n_classes),stddev=0.1),name='out_w')
            output_b=tf.Variable(tf.zeros(self.n_classes),name='out_b')
            self.logits=tf.multiply(att_output,output_w)+output_b
            self.predictions = tf.cast(tf.argmax(self.logits, 1), tf.int32)
            logits_softmax=tf.nn.softmax(self.logits,name='softmax')
            self.predictions_prob=tf.reduce_max(logits_softmax,1,name='predict_prob')

        with tf.variable_scope('loss'):
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits))

        with tf.variable_scope('accuracy'):
            correct_label=tf.cast(tf.argmax(self.input_y,1),tf.int32)
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predictions,correct_label),tf.float32))

        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver()


    def train(self,x_train,y_train,mode=None,restore=False):
        if mode!='continue':
            logging.info('Model building...')
            self.build()
            self.session.run(tf.global_variables_initializer())
        else:
            if restore:
                self.build()
                self.session.run(tf.global_variables_initializer())
                self.saver.restore(sess=self.session,save_path=self.save_path)

        logging.info('Start training :)')
        for epoch in range(self.n_epochs):
            loss_epoch,acc_epoch=[],[]
            batches=create_batch(list(zip(x_train,y_train)),batch_size=self.batch_size)
            for batch in batches:
                x_batch,y_batch=zip(*batch)
                feed_dict={self.input_x:x_train,self.input_y:y_batch,self.dropout_keep_prob:0.5}
                _,step,loss,acc=self.session.run([self.train_op,self.global_step,self.loss,self.accuracy],feed_dict)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                if step%100==0:
                    print('Epoch {}, Step {}, Loss:{}, Acc:{}'.format(epoch,step,loss,acc))
            logging.info('Epoch {}, Loss: {}, Accuracy: {}'.format(epoch,np.mean(loss_epoch),np.mean(acc_epoch)))


    def test(self,x_test,y_test,restore=False):
        if restore:
            self.build()
            self.session.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.session,save_path=self.save_path)
        test_loss,test_acc=[],[]
        batches=create_batch(list(zip(x_test,y_test)),batch_size=config.batch_size)
        for batch in batches:
            x_batch,y_batch=zip(*batch)
            feed_dict={self.input_x:x_batch, self.input_y:y_batch,self.dropout_keep_prob:1.0}
            loss,acc=self.session.run([self.loss,self.accuracy],feed_dict)
            test_acc.append(loss)
            test_acc.append(acc)
        logging.info('Test result: loss {}, accuracy {}'.format(np.mean(test_loss),np.mean(test_acc)))

    def predict(self,x_new,restore=False):
        if restore:
            self.build()
            self.session.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.session,save_path=self.save_path)

        feed_dict={self.input_x:x_new,self.dropout_keep_prob:1.0}
        pred,prob=self.session.run([self.predictions,self.predictions_prob],feed_dict)

    def plot(self):
        f = open('visualize.html', 'w')
        f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')


if '__main__' == __name__:
    x_train, x_test, y_train, y_test, word2index,n_classes,index2word,index2label,tfidf_feature=create_data(sentence_length=config.sentence_length,sample=None)
    sess = tf.Session()
    config.vocabulary_size=len(word2index)
    gru_att=Self_attention(sess)
    gru_att.train(x_train,y_train)
    gru_att.test(x_test,y_test)
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