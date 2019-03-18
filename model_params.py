from collections import defaultdict
import json

params=defaultdict(
    file_based=True,
    batch_size=128,
    num_train_epochs=5,
    learning_rate=1e-4, # original value if use warm up

    vocab_size=None,
    n_class=None,
    model_dir='./outputs',
    data_dir='./data',
    output_dir='./outputs',
    log_dir='./log',

    #embedding
    chinese_seg='word',  # word ,char, mix
    seq_length=90, #adjust

    embedding_size=300,
    embedding_type='fasttext_finetune', # random, word2vec_static,word2vec_finetune,fasttext_static,fasttext_finetune,multi_channel
    embedding_dropout_keep=0.95,

    #CNN
    kernel_sizes=[3,5,7],
    filters=128,

    # RNN model parameters
    lstm_hidden_size=512,
    gru_hidden_size=512,
    attention_hidden_size=128,
    rnn_dropout_keep=0.95,

    #Bert model parameters
    bert_config_file='./pretrained/bert_chinese/bert_config.json',
    bert_init_checkpoint='./pretrained/bert_chinese/bert_model.ckpt',
    #word2vec pretrained
    word2vec_file='./pretrained/word2vec_cn.cbow.bin',
    word2vec_file2='./pretrained/sgns.merge.word.bz2',
    fasttext_file='./pretrained/fasttext_cc.zh.300.vec.gz',

    #
    len_train_examples=None,
    len_dev_examples=None,
    len_test_examples=None,

    #
    penalization=False,

    #
    dense_hidden_size=1024,
    dropout_keep=0.95,
)


class Config(object):
    def __init__(self):
        self.params=defaultdict()

    def from_json_file(self,json_file):
        with open(json_file, 'r') as f:
            self.params = json.load(f)

    def to_json_string(self,json_file,params):
        with open(json_file, 'w') as f:
            json.dump(params, f)

config=Config()

if __name__=='__main__':
    config.to_json_string('./config.json',params)
    #config.from_json_file('./config.json')



