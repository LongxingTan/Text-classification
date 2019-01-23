import os
import csv
import tensorflow as tf
import collections
import tokenization
from run_classifier import params

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
            reader=csv.reader(f,delimiter=',',quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class OnlineProcessor(DataProcessor):
    def __init__(self,seq_lenth):
        self.labels=set()
        self.features=[]
        self.seq_length=seq_lenth
        self.tokenizer=tokenization.FullTokenizer(vocab_file=os.path.join(params['data_dir'],'vocab.txt'))

    def get_train_examples(self,data_dir):
        examples=self._create_examples(self._read_csv(os.path.join(data_dir,'train.csv')),'train')
        label_list=self.get_labels()
        params.update(n_class=len(label_list))
        self._file_based_convert_examples_to_features(examples,self.get_labels(),self.seq_length,self.tokenizer,
                                                      output_file=os.path.join(data_dir,'train.tf_record'))
        return

    def get_dev_axamples(self,data_dir):
        dev=self._create_examples(self._read_csv(os.path.join(data_dir,'dev.csv')),'dev')
        self._file_based_convert_examples_to_features(dev,self.get_labels(),self.seq_length,self.tokenizer,
                                                      output_file=os.path.join(data_dir,'dev.tf_record'))


    def get_test_examples(self,data_dir):
        test=self._create_examples(self._read_csv(os.path.join(data_dir, 'test.csv')), 'test')
        self._file_based_convert_examples_to_features(test,self.get_labels(),self.seq_length,self.tokenizer,
                                                      output_file=os.path.join(data_dir,'test.tf_record'))

    def get_labels(self):
        return list(self.labels)

    def _create_examples(self,lines,set_type):
        examples=[]
        for (i,line) in enumerate(lines):
            guid="%s-%s"%(set_type,i)
            text_a=tokenization.convert_to_unicode(line[0])
            label=tokenization.convert_to_unicode(line[1])
            self.labels.add(label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=None,label=label))
        return examples

    def _convert_single_example(self,ex_index,example,label_list,max_seq_length,tokenizer):
        label_map={}
        for i,label in enumerate(label_list):
            label_map[label]=i

        tokens_a=tokenizer.tokenize(example.text_a)
        if len(tokens_a)>max_seq_length-2:
            tokens_a=tokens_a[0:(max_seq_length-2)]

        tokens=[]
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids=tokenizer.convert_tokens_to_ids(tokens)
        while len(input_ids)<max_seq_length:
            input_ids.append(0)
        assert len(input_ids)==max_seq_length

        label_id=label_map[example.label]
        feature=InputFeatures(input_ids=input_ids,label_id=label_id)
        return feature


    def _file_based_convert_examples_to_features(self,examples,label_list,max_seq_length,tokenizer,output_file):
        writer=tf.python_io.TFRecordWriter(output_file)
        for ex_index,example in enumerate(examples):
            if ex_index%10000==0:
                tf.logging.info("writing exmaples %d of %d" %(ex_index,len(examples)))
            feature=self._convert_single_example(ex_index,example,label_list,max_seq_length,tokenizer)

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
        d=d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record,name_to_features),
            batch_size=params['batch_size'],))
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

def input_fn_builder(features,labels):
    def input_fn():
        tf.estimator.inputs.numpy_input_fn(
            x={'features':features},
            y=labels,
            batch_size=params["batch_size"],
            shuffle=False
        )

    return input_fn