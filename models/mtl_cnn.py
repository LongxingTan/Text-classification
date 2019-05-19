# encoding:utf-8
# multi task learning of CNN

import os
import csv
import collections
import tokenization
import tensorflow as tf
from models._embedding import Embedding_layer
from models._normalization import BatchNormalization,LayerNormalization
from models._loss import create_loss
from models._optimization import create_optimizer_basic_adam,create_optimizer_warmup_adam
from models._eval import create_eval,create_eval_sk
from tokenization import create_vocab_and_label


class TextCNN(object):
    def __init__(self,training,params):
        self.training=training
        self.params=params
        self.embedding_layer=Embedding_layer(vocab_size=params['vocab_size'],
                                             embed_size=params['embedding_size'],
                                             embedding_type=params['embedding_type'],
                                             params=params)
        self.bn_layer=BatchNormalization()

    def build(self,inputs):
        with tf.name_scope("embed"):
            embedded_outputs=self.embedding_layer(inputs) # => batch_size* seq_length* embedding_dim

        if self.training:
            embedded_outputs=tf.nn.dropout(embedded_outputs,self.params['embedding_dropout_keep'])

        conv_output = []
        for i, kernel_size in enumerate(self.params['kernel_sizes']):
            with tf.name_scope("conv_%s" % kernel_size):
                conv1=tf.layers.conv1d(inputs=embedded_outputs,
                                       filters=self.params['filters'],
                                       kernel_size=[kernel_size],
                                       strides=1,
                                       padding='valid',
                                       activation=tf.nn.relu) # => batch_size *(seq_length-kernel_size+1)* filters
                pool1=tf.layers.max_pooling1d(inputs=conv1,
                                              pool_size=self.params['seq_length'] - kernel_size + 1,
                                              strides=1) # => batch_size * 1 * filters
                conv_output.append(pool1)

        self.cnn_output_concat=tf.concat(conv_output,2) # => batch_size*1* (filters*len_kernels)
        self.cnn_out=tf.squeeze(self.cnn_output_concat,axis=1)
        self.cnn_out = self.bn_layer(self.cnn_out)

        if self.training:
            self.cnn_out=tf.nn.dropout(self.cnn_out,self.params['dropout_keep'])

        assert isinstance(self.params['n_class'], (list, tuple)),"For multi task learning, please set n_class as list"
        self.logits=[]
        for i,n_class_task in enumerate(self.params['n_class']):
            self.logits[i]=tf.layers.dense(self.cnn_out,units=n_class_task)


    def __call__(self,inputs,targets=None):
        self.build(inputs)
        return self.logits



class InputExample:
    def __init__(self,guid,text_a,text_b=None,label=None,label_b=None):
        self.guid=guid
        self.text_a=text_a
        self.text_b=text_b
        self.label=label
        self.label_b=label_b

class InputFeatures:
    def __init__(self,input_ids,label_ids,label_b_ids=None):
        self.input_ids=input_ids
        self.label_ids=label_ids
        self.label_b_ids=label_b_ids

class DataProcessor:
    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_examples(self,data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls,input_file,quotechar=None):
        with open(input_file,'r',encoding='utf-8') as f:
            reader=csv.reader(f,quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines



class MTL_OnlineProcessor(DataProcessor):
    def __init__(self,params,seq_length,chinese_seg,generate_label_map=False):
        self.seq_length = seq_length
        self.params = params #pass parameters by reference in python
        self.tokenizer = tokenization.BasicTokenizer(chinese_seg=chinese_seg, params=params)
        self.generate_label_map = generate_label_map
        if self.generate_label_map:
            self.labels=set(['NA'])
            self.label_map = {}
        else:
            _,self.label_map=self.load_label_dict()
            _,self.label_map_b=self.load_label_dict()


    def get_train_examples(self,data_dir,generate_file=False):
        self.train=self._create_examples(self._read_csv(os.path.join(data_dir,'train.csv')),'train')
        self.params['len_train_examples'] = len(self.train)

        if self.generate_label_map:
            for i, label in enumerate(self.get_labels()):
                self.label_map[label] = i
            self.params.update(n_class=len(self.get_labels()))

        if generate_file:
            self._file_based_convert_examples_to_features(self.train,self.seq_length,self.tokenizer,
                                                          output_file=os.path.join(data_dir,'train.tf_record'))

        else:
            train_features=self._convert_examples_to_features(self.train,self.seq_length,self.tokenizer)
            return train_features


    def get_dev_examples(self,data_dir,generate_file=False):
        dev=self._create_examples(self._read_csv(os.path.join(data_dir,'dev.csv')),'dev')
        self.params['len_dev_examples'] = len(dev)

        if generate_file:
            self._file_based_convert_examples_to_features(dev,self.seq_length,self.tokenizer,
                                                          output_file=os.path.join(data_dir,'eval.tf_record'))
        else:
            dev_features=self._convert_examples_to_features(dev,self.seq_length,self.tokenizer)
            return dev_features


    def get_test_examples(self,data_dir,generate_file=False):
        test=self._create_examples(self._read_csv(os.path.join(data_dir, 'test.csv')), 'test')
        self.params['len_test_examples'] = len(test)

        if generate_file:
            self._file_based_convert_examples_to_features(test,self.seq_length,self.tokenizer,
                                                          output_file=os.path.join(data_dir,'test.tf_record'))
        else:
            test_features = self._convert_examples_to_features(test, self.seq_length, self.tokenizer)
            return test_features

    def get_labels(self):
        return list(self.labels)

    def _create_examples(self,lines,set_type):
        # modify this while different data
        examples=[]
        for (i,line) in enumerate(lines):
            guid="%s-%s"%(set_type,i)
            text_a=tokenization.convert_to_unicode(line[0]) # note that if line[0] is the completed text
            if set_type=='test':
                label='NA'
                label_b='NA'
            else:
                label=tokenization.convert_to_unicode(line[1])
                label_b=tokenization.convert_to_unicode(line[-1])

            if set_type=='train' and self.generate_label_map:
                self.labels.add(label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=None,label=label,label_b=label_b))
        return examples


    def _convert_single_example(self,example,seq_length,tokenizer):
        tokens_a=tokenizer.tokenize(example.text_a) #Todo: optimiza here if you want char and word concat input
        if self.params['chinese_seg']=='mixed':
            tokenizer_word= tokenization.BasicTokenizer(chinese_seg='word', params=self.params)
            tokenizer_char=tokenization.BasicTokenizer(chinese_seg='char', params=self.params)
            tokens_a_word = tokenizer_word.tokenize(example.text_a)
            tokens_a_char=tokenizer_char.tokenize(example.text_a)

        if len(tokens_a)>seq_length-2:
            tokens_a=tokens_a[0:(seq_length-2)]

        tokens=[]
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids=tokenizer.convert_tokens_to_ids(tokens=tokens)
        while len(input_ids)<seq_length:
            input_ids.append(0)
        assert len(input_ids)==seq_length

        if example.label in self.label_map.keys():
            label_id=self.label_map[example.label]
            label_b_id=self.label_map_b[example.label_b]
        else:
            label_id=self.label_map['NA']
            label_b_id=self.label_map_b['NA']
        feature=InputFeatures(input_ids=input_ids,label_ids=label_id,label_b_ids=label_b_id)
        #print('ids',example.label,'tokens',tokens)
        return feature

    def _convert_examples_to_features(self,examples,seq_length,tokenizer):
        features=[]
        for i,example in enumerate(examples):
            if i%10000==0:
                tf.logging.info("process examples %d of %d" %(i,len(examples)))
            feature=self._convert_single_example(example,seq_length,tokenizer)
            features.append(feature)
        return features


    def _file_based_convert_examples_to_features(self,examples,seq_length,tokenizer,output_file):
        writer=tf.python_io.TFRecordWriter(output_file)
        for i,example in enumerate(examples):
            if i%10000==0:
                tf.logging.info("writing examples %d of %d" %(i,len(examples)))
            feature=self._convert_single_example(example,seq_length,tokenizer)

            features=collections.OrderedDict()
            features['input_ids']=tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.input_ids)))
            features['label_ids']=tf.train.Feature(int64_list=tf.train.Int64List(value=list([feature.label_ids])))

            tf_example=tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def load_label_dict(self,label_dict_dir='./data/label_dict.txt'):
        index2label, label2index = {}, {}
        reader = open(label_dict_dir, 'r').readlines()
        for row in reader:
            row=row.split(',')
            index2label.update({int(row[1]): row[0]})
            label2index.update({row[0]: int(row[1])})
        return index2label,label2index



def input_fn_builder(features,batch_size,seq_length,is_training):
    input_ids=[]
    label_ids=[]
    for feature in features:
        input_ids.append(feature.input_ids)
        label_ids.append(feature.label_id)

    def input_fn():
        num_examples=len(features)
        d=tf.data.Dataset.from_tensor_slices({
            "input_ids":tf.constant(input_ids,shape=[num_examples,seq_length],dtype=tf.int32),
            "label_ids":tf.constant(label_ids,shape=[num_examples],dtype=tf.int32)})
        if is_training:
            d=d.repeat()
            d=d.shuffle(buffer_size=100)
        d=d.batch(batch_size=batch_size)
        return d
    return input_fn


def model_fn_builder(textmodel,params,init_checkpoint=None):
    def model_fn(features, labels, mode):
        inputs, targets,targets_b = features['input_ids'], features['label_ids'],features['label_b_ids']
        model=textmodel(training=(mode==tf.estimator.ModeKeys.TRAIN),params=params)
        logits = model(inputs, targets)
        targets_onehot = tf.one_hot(targets, depth=params['n_class'],dtype=tf.float32)

        loss = create_loss(logits=logits, y_onehot=targets_onehot, loss_type='multi_task_loss')
        prediction_label = tf.argmax(logits, -1,output_type=tf.int32)
        probabilities = tf.reduce_max(tf.nn.softmax(logits, name='softmax_tensor'),axis=-1)
        #accuracy= tf.metrics.accuracy(labels=targets, predictions=prediction_label)

        correct_predictions = tf.equal(prediction_label,targets)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        logging_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": accuracy}, every_n_iter=100)
        tf.summary.scalar('accuracy', accuracy)


        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("**** Start predict ****")
            prediction_dict = {"labels": prediction_label,
                               'probabilities':probabilities}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=prediction_dict)


        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("**** Start evaluate ****")
            eval_metric_ops = create_eval(targets, prediction_label)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

        else:
            #train_op=create_optimizer_basic_adam(loss,learning_rate=params['learning_rate'])
            num_train_steps = int(params['len_train_examples'] / params['batch_size'] * params['num_train_epochs'])
            train_op=create_optimizer_warmup_adam(loss,init_learning_rate=params['learning_rate'],num_train_steps=num_train_steps,num_warmup_steps=int(0.10*num_train_steps))
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[logging_hook])
    return model_fn

if __name__=='__main__':
    'pass'


