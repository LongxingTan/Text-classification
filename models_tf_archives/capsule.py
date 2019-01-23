from models_archives.capsule_layer import *
from Capsule.config import config
from utils import *
import logging

class Capsule:
    def __init__(self,sess,config):
        self.session=sess
        self.config=config

    def build(self):
        self.input_x=tf.placeholder(tf.int32,[None,config.sentence_length],name='input_x')
        self.input_y=tf.placeholder(tf.int32,[None,config.n_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')
        self.global_step=tf.Variable(0,name='global_step')

        with tf.name_scope('embed'):
            if config.embedding_type=='random':
                w_embed=tf.Variable(tf.random_uniform([config.vocabulary_size,config.embedding_dim],-0.25,0.25),name='embed_w')
                x_embed=tf.nn.embedding_lookup(w_embed,self.input_x)
                x_embed=x_embed[...,tf.newaxis]

        with tf.name_scope('conv'):
            filter_shape=[config.conv_filter_wordsize]+[config.embeddint_dim,1,config.conv_filter_num]
            w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b_conv= tf.Variable(tf.constant(0.1, shape=[config.conv_filter_num]), name='b')
            x_conv_prev=tf.nn.conv2d(x_embed,filter=w_conv,strides=[1,1,1,1],padding='VALID')
            x_conv = tf.nn.relu(tf.nn.bias_add(x_conv_prev, b_conv), name='relu')

        with tf.name_scope('capsule_conv'):
            capsule_conv=Capsule_layer(n_vector=config.capsule_conv_vector , if_routing=False, layer_type='CONV')
            x_cap_conv=capsule_conv(x_conv, kernel_size=[1,1,config.conv_filter_num,config.capsule_conv_filter_num],
                                    stride=[1,1,1,1] )

        with tf.name_scope('capsule_fc'):
            capsule_fc=Capsule_layer(n_vector=config.capsule_fc_vector ,if_routing=True, layer_type='FC')
            x_cap_fc=capsule_fc(x_cap_conv,output_dim=config.n_classes)


        with tf.name_scope('loss'):
            loss=tf.nn.softmax_cross_entropy_with_logits(logits=x_cap_fc,labels=self.input_y)
            self.loss=tf.reduce_mean(loss)

        with tf.name_scope('accuracy'):
            correct_label=tf.cast(tf.argmax(self.input_y,axis=1),tf.int32)
            predictions=tf.cast(tf.argmax(x_cap_fc,axis=1),tf.int32)
            self.accuracy=tf.reduce_mean(tf.equal(correct_label,predictions))

        self.train_op=tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver()


    def train(self,x_train,y_train,mode=None,restore=False):
        if mode != 'continue':
            logging.info('Model building...')
            self.build()
            self.session.run(tf.global_variables_initializer())
        else:
            if restore:
                self.build()
                self.session.run(tf.global_variables_initializer())
                self.saver.restore(sess=self.session, save_path=self.config.save_path)

        logging.info('Start training :)')
        for epoch in range(self.config.n_epochs):
            loss_epoch, acc_epoch = [], []
            batches = create_batch(list(zip(x_train, y_train)), batch_size=self.config.batch_size)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {self.input_x: x_train, self.input_y: y_batch, self.dropout_keep_prob: 0.5}
                _, step, loss, acc = self.session.run([self.train_op, self.global_step, self.loss, self.accuracy],
                                                      feed_dict)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                if step % 100 == 0:
                    print('Epoch {}, Step {}, Loss:{}, Acc:{}'.format(epoch, step, loss, acc))
            logging.info('Epoch {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(loss_epoch), np.mean(acc_epoch)))

    def test(self,x_test,y_test):
        pass

    def predict(self,x_data):
        pass


if '__main__' == __name__:
    x_train, x_test, y_train, y_test, word2index,n_classes,index2word,index2label,tfidf_feature=create_data(sentence_length=config.sentence_length,sample=None)
    sess = tf.Session()
    config.vocabulary_size=len(word2index)
    cap_net=Capsule()