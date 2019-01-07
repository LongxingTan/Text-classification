#feed data use tf.placeholder and tf.Session().run()

import tensorflow as tf
from modeling import *
from config import *
from utils import *
import optimization

class Bert(object):
    def __init__(self,bert_config,num_labels,seq_length,init_checkpoint):
        self.bert_config=bert_config
        self.num_labels=num_labels
        self.seq_length=seq_length

        self.input_ids=tf.placeholder(tf.int32,[None,self.seq_length],'input_ids')
        self.input_mask=tf.placeholder(tf.int32,[None,self.seq_length],'input_mask')
        self.segment_ids=tf.placeholder(tf.int32,[None,self.seq_length],'segment_ids')
        self.labels=tf.placeholder(tf.int32,[None],name='labels')
        self.is_training=tf.placeholder(tf.bool,name='is_training')
        self.learning_rate=tf.placeholder(tf.float32,name='learning_rate')

        self.model=BertModel(self.bert_config,is_training=self.is_training,input_ids=self.input_ids,
                             input_mask=self.input_mask,token_type_ids=self.segment_ids)

        tvars=tf.trainable_variables()
        initialized_variable_names={}
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint,assignment_map)

        tf.logging.info("**trainable variables**")
        for var in tvars:
            init_string=""
            if var.name in initialized_variable_names:
                init_string=", *INIT_FROM_CKPT*"
            tf.logging.info("name=%s, sgape=%s%s",var.name,var.shape,init_string)
        self.inference()

    def inference(self):
        output_layer=self.model.get_pooled_output()

        with tf.variable_scope("loss"):
            def apply_dropout_last_layer(output_layer):
                output_layer=tf.nn.dropout(output_layer,keep_prob=0.9)
                return output_layer

            def not_apply_dropout(output_layer):
                return output_layer

            output_layer=tf.cond(self.is_training,lambda: apply_dropout_last_layer(output_layer),
                                 lambda : not_apply_dropout(output_layer))
            self.logits=tf.layers.dense(output_layer,self.num_labels,name='fc')
            self.y_pred_cls=tf.argmax(tf.nn.softmax(self.logits),1,name='pred')

            one_hot_labels=tf.one_hot(self.labels,depth=self.num_labels,dtype=tf.float32)
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=one_hot_labels)
            self.loss=tf.reduce_mean(cross_entropy,name='loss')
            self.train_op=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


        with tf.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(one_hot_labels,1),self.y_pred_cls)
            self.accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')


def make_tf_record(config):
    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        raise ValueError("Output directory already exists and is not empty")
    os.makedirs(config.output_dir,exist_ok=True)

    online_processor = OnlineProcessor()

    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
    train_file=os.path.join(config.data_dir,"train.tf_record")
    eval_file=os.path.join(config.data_dir,"eval_tf_record")

    train_examples=online_processor.get_train_examples(config.data_dir)
    label_list = online_processor.get_labels()
    config.num_labels = len(label_list)
    tf.logging.info("label size: {}".format(len(label_list)))

    file_based_convert_examples_to_features(
        train_examples, label_list, config.max_seq_length, tokenizer, train_file)

    eval_examples=online_processor.get_dev_axamples(config.data_dir)
    file_based_convert_examples_to_features(
        eval_examples, label_list, config.max_seq_length, tokenizer, eval_file)
    del train_examples,eval_examples


def _decode_record(record,name_to_features):
    example=tf.parse_single_example(record,name_to_features)
    for name in list(example.keys()):
        t=example[name]
        if t.dtype==tf.int64:
            t=tf.to_int32(t)
        example[name]=t
    return example

def read_data(data,batch_size,is_training,num_epochs):
    name_to_features={
        "input_ids":tf.FixedLenFeature([config.max_seq_length],tf.int64),
        "input_mask":tf.FixedLenFeature([config.max_seq_length],tf.int64),
        "segment_ids":tf.FixedLenFeature([config.max_seq_length],tf.int64),
        "label_ids":tf.FixedLenFeature([],tf.int64)}

    if is_training:
        data=data.shuffle(buffer_size=50000)
        data=data.repeat(num_epochs)

    data=data.apply(tf.contrib.data.map_and_batch(
        lambda record: _decode_record(record,name_to_features),
        batch_size=batch_size))
    return data

def get_test_examples():
    online_processor = OnlineProcessor()
    label_list = online_processor.get_labels()
    tf.logging.info("label size: {}".format(len(label_list)))
    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)

    examples=online_processor.get_test_examples(config.data_dir)
    features=get_test_features(examples,label_list,config.max_seq_length,tokenizer)
    return features

def evaluate(sess,model):
    test_record=tf.contrib.data.TFRecordDataset("./data/eval.tf_record")
    test_data=read_data(test_record,config.train_batch_size,False,3)
    test_iterator=test_data.make_one_shot_iterator()
    test_batch=test_iterator.get_next()

    data_nums=0
    total_loss=0.0
    total_acc=0.0
    while True:
        try:
            features=sess.run(test_batch)
            feed_dict={model.input_ids:features["input_ids"],
                       model.input_mask:features["input_mask"],
                       model.segment_ids:features["segment_ids"],
                       model.labels:features["label_ids"],
                       model.is_training:False,
                       model.learning_rate:config.learning_rate}
            batch_len=len(features["input_ids"])
            data_nums+=batch_len
            loss,acc=sess.run([model.loss,model.accuracy],feed_dict=feed_dict)
            total_loss+=loss*batch_len
            total_acc+=acc*batch_len
        except Exception as e:
            print(e)
            break
    return total_loss/data_nums,total_acc/data_nums

def main():
    bert_config = BertConfig.from_json_file(config.bert_config_file)
    with tf.Graph().as_default():
        train_record=tf.data.TFRecordDataset("./data/train.tf_record")
        train_data=read_data(train_record,config.train_batch_size,True,3)
        train_iterator=train_data.make_one_shot_iterator()

        model=Bert(bert_config=bert_config,num_labels=config.num_labels,seq_length=config.max_seq_length,init_checkpoint=config.init_checkpoint)
        sess=tf.Session()
        saver=tf.train.Saver()

        train_step=0
        val_loss=0.0
        val_acc=0.0
        best_val_acc=0.0

        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            train_batch=train_iterator.get_next()
            while True:
                try:
                    train_step+=1
                    features=sess.run(train_batch)
                    feed_dict={model.input_ids:features['input_ids'],
                               model.input_mask:features["input_mask"],
                               model.segment_ids:features["segment_ids"],
                               model.labels:features['label_ids'],
                               model.is_training:True,
                               model.learning_rate:config.learning_rate}
                    _,train_loss,train_acc=sess.run([model.train_op,model.loss,model.accuracy],feed_dict=feed_dict)

                    if train_step%100==0:
                        val_loss,val_acc=evaluate(sess,model)

                    if val_acc>best_val_acc:
                        best_val_acc=val_acc
                        saver.save(sess,"./model/bert/model",global_step=train_step)
                        improved_str="*"
                    else:
                        improved_str=''

                    tf.logging.info("Steps {}, train_loss {},train_acc {},val_loss {},val_acc {}".format(
                        train_step,train_loss,train_acc,val_loss,val_acc))
                except Exception as e:
                    print(e)
                    break



if __name__=="__main__":
    make_tf_record(config)
    main()

