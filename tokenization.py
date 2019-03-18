# encoding:utf-8
import unicodedata
import jieba
#import snownlp
#import thulac
#from nltk.tokenize.stanford_segmenter import StanfordSegmenter
#import HanLP
import collections
import logging
import os

def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError("unsupported text string type: %s" % (type(text)))


def create_vocab_and_label(params):
    from prepare_inputs import OnlineProcessor
    online = OnlineProcessor(params=params, seq_length=params["seq_length"], chinese_seg=params['chinese_seg'],generate_label_map=True)
    online.get_train_examples(data_dir=params['data_dir'])

    if params['chinese_seg']=='word':
        logging.info('Vocab generating ...')
        vocab = set()
        vocab.update(
            ['[PAD]', '[SEP]', '[CLS]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]'])
        tokenizer = BasicTokenizer(chinese_seg='word', params=params)
        for example in online.train:
            vocab.update(tokenizer.tokenize(example.text_a))

        with open('./data/vocab_word.txt', 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write('%s\n' % word)
        f.close()
    else:
        logging.warning("Bert already provide a chinese char list, so will not generate new")

    with open('./data/label_dict.txt', 'w') as f:
        for key in online.label_map.keys():
            if key!='':
                f.write("%s,%s\n" % (key, online.label_map[key]))
    f.close()


def load_vocab(vocab_file, params):
    vocab=collections.OrderedDict()
    index_vocab=collections.OrderedDict()
    index=0

    with open(vocab_file,'rb') as reader:
        while True:
            tmp=reader.readline()
            token=convert_to_unicode(tmp)

            if not token:
                break

            token=token.strip()
            vocab[token]=index
            index_vocab[index]=token
            index+=1
        params.update(vocab_size=len(vocab))
    return vocab,index_vocab


class BasicTokenizer(object):
    def __init__(self,params,chinese_seg,do_lower_case=True):
        self.do_lower_case=do_lower_case
        self.chinese_seg=chinese_seg
        self.params=params
        if self.chinese_seg=='char':
            self.vocab, self.index_vocab = load_vocab(vocab_file=os.path.join(self.params['data_dir'],'vocab.txt'),
                                                      params=self.params)
        elif self.chinese_seg=='word':
            self.vocab, self.index_vocab = load_vocab(vocab_file=os.path.join(self.params['data_dir'],'vocab_word.txt'),
                                                      params=self.params)
        self.stopwords=[line.strip() for line in open('./data/stop_words.txt','r',encoding='utf-8').readlines()]


    def tokenize(self,text):
        text=convert_to_unicode(text)
        text=self._clean_text(text)

        if self.chinese_seg=="char":
            text=self._tokenize_chinese_chars(text)
        elif self.chinese_seg=="word":
            text=self._tokenize_chinese_words(text)

        orig_tokens=self._whitespace_tokenize(text)
        split_tokens=[]
        for token in orig_tokens:
            if self.do_lower_case:
                token=token.lower()
                token=self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens=self._whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def convert_tokens_to_ids(self,tokens):
        ids = []
        for token in tokens:
            if token in self.vocab.keys():
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[unused1]'])
        return ids

    def _clean_text(self,text):
        output=[]
        for char in text:
            cp=ord(char)
            if cp==0 or cp==0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self,text):
        output=[]
        for char in text:
            cp=ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_words(self,text,shorter=True):
        jieba.load_userdict('./data/user_dict.txt')
        if shorter:
            wordlist=self.shorter_chinese_cut(text)
        else:
            wordlist = jieba.cut(text,HMM=False)

        #self.vocab.update([i for i in ])
        #print("/".join(jieba.cut(text)))
        wordlist_new =[]
        for word in wordlist:
            if word not in self.stopwords:
                wordlist_new.append(word)
            else:
                continue
        return " ".join(wordlist_new)


    def _whitespace_tokenize(self,text):
        text=text.strip()
        if not text:
            return []
        tokens=text.split()
        return tokens

    def _run_strip_accents(self,text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self,text):
        chars=list(text)
        i=0
        start_new_word=True
        output=[]
        while i<len(chars):
            char=chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word=True
            else:
                if start_new_word:
                    output.append([])
                start_new_word=False
                output[-1].append(char)
            i+=1
        return ["".join(x) for x in output]


    def _is_control(self,char):
        if char=='\t' or char=='\n' or char=='\r':
            return False
        cat=unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _is_whitespace(self,char):
        if char=='\t' or char=='\n' or char=='\r':
            return True
        cat=unicodedata.category(char)
        if cat=="Zs":
            return True
        return False

    def _is_punctuation(self,char):
        cp=ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_chinese_char(self,cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True
        return False

    def shorter_chinese_cut(self,line):
        result = []
        for long_word in jieba.lcut(line, HMM=False):
            cp = ord(long_word[0])
            if self._is_chinese_char(cp) and len(long_word)>3:
                jieba.del_word(long_word)
                result.extend(jieba.lcut(long_word))
            else:
                result.append(long_word)
        return result


if __name__=="__main__":
    from model_params import params
    tokenizer=BasicTokenizer(chinese_seg='word',params=params)
    words=tokenizer._tokenize_chinese_words("客户三个工作日外第五次投诉：针对其反映的车辆，因车辆在行驶中及冷车启动时出现异响问题送修至北京保利星徽。经销商对车辆检测后告知需要更换助力泵。客户非常不满意，质疑产品质量。另外反映车辆还存在异味问题。客户对此不能接受，要求厂家回复一事，至今没有工作人员与其联系。客户再次致电表示希望由厂家给予解答，助力泵是不是两年或三年就会坏的，还是质量就有问题，客户希望得到专业的解答，其次客户希望由北京奔驰给予合理的解决方案，同时客户还表示投诉反馈的等待时间太长，不能接受，服务水平问题。现在客户非常着急，催促厂家的工作人员核实情况后尽快给予回复解决")
    result=tokenizer.shorter_chinese_cut("客户三个工作日外第五次投诉：针对其反映的车辆，因车辆在行驶中及冷车启动时出现异响问题送修至北京保利星徽。经销商对车辆检测后告知需要更换助力泵。客户非常不满意，质疑产品质量。另外反映车辆还存在异味问题。客户对此不能接受，要求厂家回复一事，至今没有工作人员与其联系。客户再次致电表示希望由厂家给予解答，助力泵是不是两年或三年就会坏的，还是质量就有问题，客户希望得到专业的解答，其次客户希望由北京奔驰给予合理的解决方案，同时客户还表示投诉反馈的等待时间太长，不能接受，服务水平问题。现在客户非常着急，催促厂家的工作人员核实情况后尽快给予回复解决")
    print(words)
    print(result)