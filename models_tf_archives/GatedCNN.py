
import tensorflow as tf
import datetime
from utils import *
from Convolutional.config import *

class GatedCNN():
    def __init__(self,sentence_length,n_classes,vocabulary_size,embedding_dim, embedding_matrix,n_layers,
                 conv_multi_channel,filter_sizes,filter_num):
        self.input_x=tf.placeholder(shape=(None,sentence_length),dtype=tf.int32,name='input_x')
        self.input_y=tf.placeholder(shape=(None,n_classes),dtype=tf.int32,name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout')

        with tf.variable_scope("embed"):
            if embedding_matrix is None:
                W_emb=tf.Variable(tf.random_uniform([vocabulary_size,embedding_dim],-1.0,1.0),name='embedding')
            else:
                W_emb=tf.get_variable(shape=[vocabulary_size,embedding_dim],
                                      initializer=tf.constant_initializer(embedding_matrix),trainable=True,name='W_emb')
            self.embed=tf.nn.embedding_lookup(W_emb,self.input_x)
            self.embed_expanded = tf.expand_dims(self.embed, -1)
            #mask_layer = np.ones((conf.batch_size, conf.context_size - 1, conf.embedding_size))
            #mask_layer[:, 0:int(conf.filter_h / 2), :] = 0
            #embed *= mask_layer
            #embed_shape = embed.get_shape().as_list()
            #embed = tf.reshape(embed, (embed_shape[0], embed_shape[1], embed_shape[2], 1))

        for i in range(n_layers):
            if not conv_multi_channel:
                #if not multi channel CNN, then the filter_sizes list indices are used for different layers
                with tf.variable_scope('layer_%d'%i):
                    assert n_layers==len(filter_sizes),"Do you really understand how to set the filter sizes here"
                    conv_linear=self.conv_op(self.embed_expanded,shape=(filter_sizes[i],embedding_dim,1,filter_num),name='linear')
                    conv_gated=self.conv_op(self.embed_expanded,shape=(filter_sizes[i],embedding_dim,1,filter_num),name='gated')
                    h=conv_linear*tf.sigmoid(conv_gated)

            else:
                #if multi channel CNN, the each channel share the same filter_sizes as different channel filters
                all_h=[]
                for filter_size in filter_sizes:
                    with tf.variable_scope('conv_%d_%d'%(i,filter_size)):
                        conv_linear=self.conv_op(self.embed_expanded,shape=(filter_size,embedding_dim,1,filter_num),filter_size=filter_size,name='linear')
                        conv_gated=self.conv_op(self.embed_expanded,shape=(filter_size,embedding_dim,1,filter_num),filter_size=filter_size,name='gated')
                        sub_h=conv_linear*tf.sigmoid(conv_gated)
                        all_h.append(sub_h)
                h=tf.concat(all_h,axis=3)
           #sample_shape = self.input_y.get_shape().as_list()
        num_filters_total = filter_num * len(filter_sizes)
        h=tf.reshape(h,(-1,num_filters_total))

        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(h,self.dropout_keep_prob)

        with tf.name_scope("output"):
            W_out=tf.get_variable("W_out",shape=[h.get_shape()[1],n_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_out=tf.Variable(tf.constant(0.1,shape=[n_classes]),name='b_out')
            self.scores=tf.nn.xw_plus_b(self.h_drop,W_out,b_out,name='scores')
            self.predictions=tf.argmax(self.scores,1,name='predictions')

        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name='accuracy')

        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')


    def conv_op(self,inputs,shape,filter_size,name):
        W=tf.get_variable('%s_W'%name,shape,tf.float32,tf.random_normal_initializer(0.0,0.1))
        b=tf.get_variable('%s_b'%name,shape[-1],tf.float32,tf.constant_initializer(1.0))
        conv=tf.nn.bias_add(tf.nn.conv2d(inputs,W,strides=[1,1,1,1],padding='VALID'),b)
        return tf.nn.max_pool(conv, ksize=[1,sentence_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                            padding='VALID', name='pool')

def GatedCNN_train(x_train, y_train, x_test, y_test,sentence_length, vocabulary_size, embedding_dim,
                  filter_sizes,filter_num, n_classes, embedding_matrix,n_layers,
                 conv_multi_channel,batch_size=64,n_epochs=25,l2_reg=0.0):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            gatedcnn = GatedCNN(sentence_length=sentence_length, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
                          filter_sizes=filter_sizes, filter_num=filter_num, n_classes=n_classes,
                          embedding_matrix=embedding_matrix,n_layers=n_layers,conv_multi_channel=conv_multi_channel)
            global_step = tf.Variable(0, name='global_stop', trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(gatedcnn.loss, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables())

            best_accuracy = 0
            def test_step(x_test,y_test):
                n_test = len(x_test)
                test_accuracy = 0
                for offset in range(0, n_test, batch_size):
                    batch_x_test = x_test[offset:offset + batch_size]
                    batch_y_test = y_test[offset:offset + batch_size]
                    test_accuracy_batch = sess.run(gatedcnn.accuracy,
                                                   feed_dict={gatedcnn.input_x: batch_x_test, gatedcnn.input_y: batch_y_test,
                                                              gatedcnn.dropout_keep_prob: 1.0})
                    test_accuracy += (test_accuracy_batch * len(batch_x_test))
                test_acc = test_accuracy / n_test
                return test_acc

            sess.run(tf.global_variables_initializer())
            batches = create_batch(list(zip(x_train,y_train)), batch_size=batch_size, n_epochs=n_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {gatedcnn.input_x: x_batch,
                             gatedcnn.input_y: y_batch,
                             gatedcnn.dropout_keep_prob: 0.5}
                _, step, loss, accuracy = sess.run([train_op, global_step, gatedcnn.loss, gatedcnn.accuracy], feed_dict)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % 10 == 0:
                    best_at_step = test_step(x_test,y_test)
                    if best_accuracy<best_at_step:
                        saver.save(sess, './tf_gatedcnn.ckpt')
                        best_accuracy=best_at_step
                        time_str = datetime.datetime.now().isoformat()
                        logging.critical("{}: step {}, loss {:g}, acc {:g},test_acc {:g}".format(time_str, step, loss, accuracy, best_accuracy))


if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('CAC.csv')
    #x_data, y_data = x_data[:100], y_data[:100] ##delete
    #n_classes = len(np.unique(y_data)) ##delete
    x_token = x_data.apply(tokenize_sentence)
    sentence_padded, sentence_length=pad_sentence(x_token)
    x_token_pad, sentence_length = pad_sentence(x_token,config.sentence_length)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    #embedding_matrix = word_embedding(word2index)
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)

    GatedCNN_train(x_train, y_train, x_test, y_test,sentence_length, vocabulary_size=len(word2index), embedding_dim=300,
                  filter_sizes=[2, 4, 6], filter_num=64, n_classes=n_classes,n_layers=1,conv_multi_channel=True,
                  embedding_matrix=None, batch_size=64,n_epochs=20)


#paper refernece
#https://arxiv.org/abs/1612.08083
#code reference:
#https://github.com/anantzoid/Language-Modeling-GatedCNN
#https://github.com/wabyking/Gated_CNN_for_language_Modeling/blob/master/GCNN.py
