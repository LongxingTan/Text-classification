import os
import csv
import copy
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
    def __init__(self,input_ids,label_id):
        self.input_ids=input_ids
        self.label_id=label_id

class DataProcessor:
    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_examples(self,data_dir):
        raise NotImplementedError()

    def get_labels(self):
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
    def __init__(self,params,seq_length,chinese_seg,file_based=False):
        self.seq_length = params['seq_length']
        self.file_based = params['file_based']
        self.params=params #pass parameters by reference in python
        self.labels=set()

        if params['chinese_seg']=='char':
            self.tokenizer=tokenization.BasicTokenizer(chinese_seg='char',params=params)
        elif params['chinese_seg']=='word':
            self.tokenizer=tokenization.BasicTokenizer(chinese_seg='word',params=params)

    def get_train_examples(self,data_dir):
        self.examples=self._create_examples(self._read_csv(os.path.join(data_dir,'train.csv')),'train')
        self.params['len_train_examples'] = len(self.examples)
        label_list=self.get_labels()
        self.params.update(n_class=len(label_list))

        self.label_map = {}
        for i, label in enumerate(label_list):
            self.label_map[label] = i

        with open('label_dict.csv', 'w') as f:
            for key in self.label_map.keys():
                f.write("%s,%s\n" % (key, self.label_map[key]))

        if self.file_based:
            self._file_based_convert_examples_to_features(self.examples,self.get_labels(),self.seq_length,self.tokenizer,
                                                          output_file=os.path.join(data_dir,'train.tf_record'))

        else:
            train_features=self.convert_examples_to_features(self.examples,self.seq_length,self.tokenizer)
            return train_features


    def get_dev_examples(self,data_dir):
        dev=self._create_examples(self._read_csv(os.path.join(data_dir,'dev.csv')),'dev')
        self.params['len_dev_examples'] = len(dev)

        if self.file_based:
            self._file_based_convert_examples_to_features(dev,self.get_labels(),self.seq_length,self.tokenizer,
                                                          output_file=os.path.join(data_dir,'eval.tf_record'))
        else:
            dev_features=self.convert_examples_to_features(dev,self.seq_length,self.tokenizer)
            return dev_features


    def get_test_examples(self,data_dir):
        test=self._create_examples(self._read_csv(os.path.join(data_dir, 'test.csv')), 'test')
        self.params['len_test_examples'] = len(test)

        if self.file_based:
            self._file_based_convert_examples_to_features(test,self.get_labels(),self.seq_length,self.tokenizer,
                                                          output_file=os.path.join(data_dir,'test.tf_record'))
        else:
            test_features = self.convert_examples_to_features(test, self.seq_length, self.tokenizer)
            return test_features

    def get_labels(self):
        return list(self.labels)

    def _create_examples(self,lines,set_type):
        examples=[]
        for (i,line) in enumerate(lines):
            guid="%s-%s"%(set_type,i)
            text_a=tokenization.convert_to_unicode(line[0])

            if set_type=='test':
                label='0'
            else:
                label=tokenization.convert_to_unicode(line[-1])

            if set_type=='train':
                self.labels.add(label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=None,label=label))
        return examples


    def _convert_single_example(self,example,seq_length,tokenizer):
        tokens_a=tokenizer.tokenize(example.text_a)
        if len(tokens_a)>seq_length-2:
            tokens_a=tokens_a[0:(seq_length-2)]

        tokens=[]
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids=tokenizer.convert_tokens_to_ids(vocab_file=os.path.join(self.params['data_dir'],'vocab_word.txt'),tokens=tokens)
        while len(input_ids)<seq_length:
            input_ids.append(0)
        assert len(input_ids)==seq_length

        if example.label in self.label_map.keys():
            label_id=self.label_map[example.label]
        else:
            label_id=0 #Todo
        feature=InputFeatures(input_ids=input_ids,label_id=label_id)
        #print('ids',example.label,'tokens',tokens)
        return feature

    def convert_examples_to_features(self,examples,seq_length,tokenizer):
        features=[]
        for i,example in enumerate(examples):
            if i%10000==0:
                tf.logging.info("process exmaples %d of %d" %(i,len(examples)))
            feature=self._convert_single_example(example,seq_length,tokenizer)
            features.append(feature)
        return features


    def _file_based_convert_examples_to_features(self,examples,label_list,seq_length,tokenizer,output_file):
        writer=tf.python_io.TFRecordWriter(output_file)
        for ex_index,example in enumerate(examples):
            if ex_index%10000==0:
                tf.logging.info("writing exmaples %d of %d" %(ex_index,len(examples)))
            feature=self._convert_single_example(example,seq_length,tokenizer)

            features=collections.OrderedDict()
            features['input_ids']=tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.input_ids)))
            features['label_ids']=tf.train.Feature(int64_list=tf.train.Int64List(value=list([feature.label_id])))

            tf_example=tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


def file_based_input_fn_builder(input_file,is_training,params):
    name_to_features={"input_ids":tf.FixedLenFeature([params['seq_length']],tf.int64),
                      "label_ids":tf.FixedLenFeature([],tf.int64)}
    def input_fn():
        d=tf.data.TFRecordDataset(input_file)
        if is_training:
            d=d.repeat()
            d=d.shuffle(buffer_size=100)
        d=d.apply(tf.contrib.data.map_and_batch(
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

def input_fn_builder(features,labels,batch_size,seq_length,is_training):
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


if __name__=="__main__":
    online = OnlineProcessor()
    train = online.get_train_examples()
