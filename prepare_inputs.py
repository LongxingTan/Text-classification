# encoding:utf-8
import os
import csv
import tensorflow as tf
import collections
import tokenization


class InputExample:
    def __init__(self,guid,text_a,text_b=None,label=None):
        self.guid=guid
        self.text_a=text_a
        self.text_b=text_b
        self.label=label


class InputFeatures:
    def __init__(self,input_ids,label_ids):
        self.input_ids=input_ids
        self.label_ids=label_ids


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


class OnlineProcessor(DataProcessor):
    def __init__(self,params,seq_length,chinese_seg,generate_label_map=False):
        self.seq_length = seq_length
        self.params = params  # pass parameters by reference in python
        self.tokenizer = tokenization.BasicTokenizer(chinese_seg=chinese_seg, params=params)
        self.generate_label_map = generate_label_map
        if self.generate_label_map:
            self.labels=set(['NA']) # add a NA to saved label_dict.txt
            self.label_map = {}
        else:
            _, self.label_map=self.load_label_dict()

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
            else:
                label=tokenization.convert_to_unicode(line[-1])
            if set_type=='train' and self.generate_label_map:
                self.labels.add(label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=None,label=label))
        return examples

    def _convert_single_example(self,example,seq_length,tokenizer):
        tokens_a=tokenizer.tokenize(example.text_a)  # Todo: optimize here if you want char and word concat input
        if self.params['chinese_seg']=='mixed':
            tokenizer_word= tokenization.BasicTokenizer(chinese_seg='word', params=self.params)
            tokenizer_char=tokenization.BasicTokenizer(chinese_seg='char', params=self.params)
            tokens_a_word = tokenizer_word.tokenize(example.text_a)
            tokens_a_char=tokenizer_char.tokenize(example.text_a)

        if len(tokens_a)>seq_length-2:
            tokens_a=tokens_a[0:(seq_length-2)]

        tokens = []
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
        else:
            label_id=self.label_map['NA']
        feature=InputFeatures(input_ids=input_ids,label_ids=label_id)
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

    def load_label_dict(self):
        index2label, label2index = {}, {}
        reader = open('./data/label_dict.txt', 'r').readlines()
        for row in reader:
            row=row.split(',')
            index2label.update({int(row[1]): row[0]})
            label2index.update({row[0]: int(row[1])})
        return index2label,label2index


def file_based_input_fn_builder(input_file,is_training,params):
    name_to_features={"input_ids":tf.FixedLenFeature([params['seq_length']],tf.int64),
                      "label_ids":tf.FixedLenFeature([],tf.int64)}
    def input_fn():
        d=tf.data.TFRecordDataset(input_file)
        if is_training:
            d=d.repeat()
            d=d.shuffle(buffer_size=100)
        d=d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record,name_to_features),
            batch_size=params['batch_size']))
        return d

    def _decode_record(record,name_to_features):
        example=tf.parse_single_example(record,name_to_features)
        for name in list(example.keys()):
            t=example[name]
            if t.dtype==tf.int64:
                t=tf.to_int32(t)
            example[name]=t
        return example
    return input_fn


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


def serving_input_receiver_fn():
    # This is used to define inputs to serve the model Todo: update
    receiver_tensors={"input_x": tf.placeholder(dtype=tf.float32, shape=[None, 5, 109], name='input_x')}
    features = {"input_x": receiver_tensors["input_x"],
                'input_y': tf.placeholder(dtype=tf.float32, shape=[None,], name='input_y')}
    return tf.estimator.export.ServingInputReceiver(features=features,receiver_tensors=receiver_tensors)
