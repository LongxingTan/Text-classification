import tokenization
import os

class InputExample:
    def __init__(self,guid,text_a,text_b=None,label=None):
        self.guid=guid
        self.text_a=text_a
        self.text_b=text_b
        self.label=label

class InputFeatures:
    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids=input_ids
        self.input_mask=input_mask
        self.segment_ids=segment_ids
        self.label_id=label_id


class DataProcessor:
    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_axamples(self,data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls,input_file,quotechar=None):
        file_in=open(input_file,"rb")
        lines=[]
        for line in file_in:
            lines.append(line.decode('gbk').split(';'))
        return lines


class OnlineProcessor(DataProcessor):
    def __init__(self):
        self.labels=set()

    def get_train_examples(self,data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir,'train.csv')),'train')

    def get_dev_axamples(self,data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir,'dev.csv')),'dev')

    def get_labels(self):
        return list(self.labels)

    def _create_examples(self,lines,set_type):
        examples=[]
        for (i,line) in enumerate(lines):
            guid="%s-%s"%(set_type,i)
            text_a=tokenization.convert_to_unicode(line[1])
            label=tokenization.convert_to_unicode(line[0])
            self.labels.add(label)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=None,label=label))
        return examples

