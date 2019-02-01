import tensorflow as tf
import sklearn as sk
import datetime
from models_tf_archives._utils import *



class Config():
    def __init__(self):
        self.embeddig_dim=300
        self.embedding_matrix=False   #False or true
        self.sentence_length=50
        self.filter_sizes=[2,4,6]
        self.filter_num=16
        self.learning_rate=10e-3
        self.batch_size=128
        self.nepochs=17
        self.chi_k=500 #chi-square top k
        self.tfidf=False

config = Config()

class textCNN():
    '''model by tensorflow, refer to https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py'''
    def __init__(self, sentence_length, vocabulary_size, embedding_dim, filter_sizes,filter_num, n_classes, embedding_matrix,l2_reg,tfidf_feature):
        self.input_x=tf.placeholder(tf.int32,[None,sentence_length],name='input_x')
        self.input_y=tf.placeholder(tf.int32,[None, n_classes], name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        self.vocabulary_size=vocabulary_size
        self.embedding_matrix=embedding_matrix
        self.embedding_dim=embedding_dim
        self.tfidf_feature = tfidf_feature
        # embedding layer
        self.embedding_chars_expanded=tf.cast(self.word_embed(),tf.float32,name='embedding_chars_sxpanded')
        if config.tfidf==True:
            self.tfidf_chars_expanded=tf.cast(self.word_tfidf(),tf.float32,name='tfidf_chars_expanded')
            self.embedding_chars_expanded=tf.concat([self.embedding_chars_expanded,self.tfidf_chars_expanded],2) #[None,sentence_length,300+500,1]

        #convolutional layer
        pooled_output=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape=[filter_size,self.embedding_chars_expanded.get_shape().as_list()[2],1,filter_num]
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b=tf.Variable(tf.constant(0.1,shape=[filter_num]),name='b')
                conv=tf.nn.conv2d(self.embedding_chars_expanded,W,strides=[1,1,1,1],padding='VALID',name='conv')
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                #pooling filter depends on the padding style
                pooled=tf.nn.max_pool(h,ksize=[1,sentence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_output.append(pooled)

        num_filters_total=filter_num*len(filter_sizes)
        self.h_pool=tf.concat(pooled_output,3)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

        #dropout layer
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        with tf.name_scope('BN'):
            self.h_bn=self.batch_normalization(self.h_drop)

        with tf.name_scope("output"):
            W_out=tf.get_variable("W_out",shape=[num_filters_total,n_classes], initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[n_classes]),name='b')
            l2_loss+=tf.nn.l2_loss(W_out)
            l2_loss+=tf.nn.l2_loss(b)
            self.scores=tf.nn.xw_plus_b(self.h_bn,W_out,b,name='scores')
            self.predictions=tf.argmax(self.scores,1,name='predictions')
            #self.predictions_prob = tf.reduce_max(self.scores, 1, name='predictions_prob')
            self.softmax=tf.nn.softmax(self.scores,name='softmax')
            self.predict_prob=tf.reduce_max(self.softmax,1,name='predict_prob')

        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg*l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name='accuracy')

    def word_embed(self):
        with tf.name_scope('word2vec'):
            if self.embedding_matrix is False:
                self.W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dim], -1.0, 1.0), name='embedding_W')
            else:
                embedding_matrix = word_embed_trans(word2index,'word2vec')
                self.W = tf.get_variable(name="W", shape=[self.vocabulary_size, self.embedding_dim],
                                         initializer=tf.constant_initializer(embedding_matrix), trainable=True)
            embedding_chars=tf.nn.embedding_lookup(self.W,self.input_x)  #[None,sentence_length,embedding_dim]
            embedding_chars_expanded = tf.expand_dims(embedding_chars, -1)
        return embedding_chars_expanded

    def word_tfidf(self):
        tfidf_matrix=use_tfidf_feature(word2index,self.tfidf_feature)
        tfidf_chars=tf.nn.embedding_lookup(tfidf_matrix,self.input_x)
        tfidf_chars_expanded=tf.expand_dims(tfidf_chars,-1)
        return tfidf_chars_expanded

    def batch_normalization(self,inputs,epsilon=1e-8,scope='batch_instance_norm'):
        with tf.variable_scope(scope):
            inputs_shape=inputs.get_shape()
            param_shape=inputs_shape[-1:]

            mean,variance=tf.nn.moments(inputs,[-1],keep_dims=True)
            beta=tf.Variable(tf.zeros(param_shape))
            gamma=tf.Variable(tf.ones(param_shape))
            normalized=(inputs-mean)/((variance+epsilon)**(0.5))
            outputs=gamma * normalized + beta
        return outputs


def textCNN_train(x_train, y_train, x_test, y_test,sentence_length, vocabulary_size, embedding_dim,
                  filter_sizes,filter_num, n_classes, embedding_matrix,batch_size,n_epochs,tfidf_feature,l2_reg=0.0):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = textCNN(sentence_length=sentence_length, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
                          filter_sizes=filter_sizes, filter_num=filter_num, n_classes=n_classes,
                          embedding_matrix=embedding_matrix,tfidf_feature=tfidf_feature, l2_reg=l2_reg)
            global_step = tf.Variable(0, name='global_stop', trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cnn.loss, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables())

            best_accuracy = 0
            def test_step(x_test,y_test):
                n_test = len(x_test)
                test_accuracy = 0
                for offset in range(0, n_test, batch_size):
                    batch_x_test = x_test[offset:offset + batch_size]
                    batch_y_test = y_test[offset:offset + batch_size]
                    test_accuracy_batch = sess.run(cnn.accuracy,
                                                   feed_dict={cnn.input_x: batch_x_test, cnn.input_y: batch_y_test,
                                                              cnn.dropout_keep_prob: 1.0})
                    test_accuracy += (test_accuracy_batch * len(batch_x_test))
                test_acc = test_accuracy / n_test
                return test_acc

            sess.run(tf.global_variables_initializer())
            batches = create_batch(list(zip(x_train,y_train)), batch_size=batch_size, n_epochs=n_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {cnn.input_x: x_batch,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 0.5}
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % 10 == 0:
                    best_at_step = test_step(x_test,y_test)
                    if best_accuracy<best_at_step:
                        saver.save(sess, './tf_cnnnet.ckpt')
                        best_accuracy=best_at_step
                        time_str = datetime.datetime.now().isoformat()
                        logging.critical("{}: step {}, loss {:g}, acc {:g},test_acc {:g}".format(time_str, step, loss, accuracy, best_accuracy))

            #test_acc=test_step(x_test, y_test)
            #logging.critical('Accuracy on test set is {} '.format(test_acc))

def textCNN_predict_new_data(x_new_data):
    #checkpoint_file=tf.train.latest_checkpoint('./tf_cnnnet.ckpt')
    checkpoint_file='./tf_cnnnet.ckpt'
    logging.critical('Loading the trained model:{}'.format(checkpoint_file))

    graph=tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver=tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess,checkpoint_file)
            input_x=graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob=graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions=graph.get_operation_by_name("output/predictions").outputs[0]
            predict_prob = graph.get_operation_by_name("output/predict_prob").outputs[0]

            batches=create_batch(x_new_data,batch_size=64,n_epochs=1,shuffle=False)
            y_new_index,y_new_prob=[],[]
            for x_test_batch in batches:
                batch_predictions,batch_predictions_prob=sess.run([predictions,predict_prob],{input_x:x_test_batch,dropout_keep_prob:1.0})
                y_new_index=np.concatenate([y_new_index,batch_predictions])
                y_new_prob=np.concatenate([y_new_prob,batch_predictions_prob],axis=0)
    return y_new_index,y_new_prob

if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('../data_raw/CAC.csv')
    #x_data, y_data = x_data[:100], y_data[:100] ##delete
    #n_classes = len(np.unique(y_data)) ##delete
    x_token = x_data.apply(tokenize_sentence)
    #sentence_padded, sentence_length=pad_sentence(x_token,sentence_length=config.sentence_length)
    x_token_pad, sentence_length = pad_sentence(x_token,sentence_length=config.sentence_length)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)


    #tfidf weight
    #x_train_label=[[index2word[word] for word in sentence] for sentence in x_train]
    x_countvec, countvec = countvectorize(x_token)
    x_vec_chi, vocab_chi = select_feature_chi(x_countvec, label2index, y_data, countvec, k=config.chi_k, if_print=False)
    x_vec_chi_tfidf, tfidf, feature = tfidf_weight(x_token, vocab_chi, if_save=False)
    tfidf_feature = dict(feature)


    #CNN
    textCNN_train(x_train, y_train, x_test, y_test,sentence_length, vocabulary_size=len(word2index), embedding_dim=config.embeddig_dim,
                  filter_sizes=config.filter_sizes, filter_num=config.filter_num, n_classes=n_classes,
                  embedding_matrix=config.embedding_matrix, batch_size=config.batch_size,n_epochs=config.nepochs,l2_reg=0.0,tfidf_feature=False)

    y_new_index,y_new_prob=textCNN_predict_new_data(x_test)
    y_new_label = [index2label[i] for i in y_new_index]
    y_test_index = np.argmax(y_test, 1)
    y_test_label = [index2label[i] for i in y_test_index]
    x_test_label = [[index2word[i] for i in sentence] for sentence in x_test]
    with open('cnn_result.csv', 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file, delimiter=';')
        w.writerow(('Description','Words', 'Label', 'Label_predicted','Probability'))
        w.writerows(zip(x_test,x_test_label, y_test_label, y_new_label,y_new_prob))

    f1=sk.metrics.f1_score(y_test_index,y_new_index,average='micro')
    confusion_matrix=sk.metrics.confusion_matrix(y_test_label,y_new_label)
    print('F1 score:',f1)
    print(confusion_matrix)

