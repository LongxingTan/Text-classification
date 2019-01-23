import tensorflow as tf
from models._embedding import Embedding_layer
from models._model_params import *


class HAN(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params
        self.embedding_layer = Embedding_layer(params['vocab_size'], params['embedding_size'])

    def build(self, inputs):
        with tf.name_scope('embed'):
            embedding_outputs = self.embedding_layer(inputs)

        if self.train:
            embedding_outputs = tf.nn.dropout(embedding_outputs, 1.0)
        pass

    def __call__(self, inputs, targets=None):
        self.build(inputs)
        return self.logits







class HAN():
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
                embedding_matrix = word_embed(word2index, self.embedding_type)
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

