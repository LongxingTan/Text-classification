from collections import defaultdict
import json

params=defaultdict(
    file_based=True,
    batch_size=128,
    num_train_epochs=1,
    learning_rate=10e-4,#5e-5
    learning_rate_warmup_steps=500,
    optimizer_adam_beta1=0.1,
    optimizer_adam_beta2=0.1,
    optimizer_adam_epsilon=10e-6,

    vocab_size=None,
    n_class=None,
    model_dir='./outputs',
    data_dir='./data',
    output_dir='./outputs',
    log_dir='./log',

    #embedding
    chinese_seg='char',  # word ,char, mix
    seq_length=100, #adjust also according to chinese_seg

    embedding_size=300,
    embedding_type='random', # random, word2vec_static,word2vec_finetune,fasttext_static,fasttext_finetune,multi_channel
    embedding_dropout_keep=0.95,

    #CNN
    kernel_sizes=[3,5,10],
    filters=128,

    # RNN model parameters
    lstm_hidden_size=128,
    gru_hidden_size=128,
    attention_hidden_size=64,
    rnn_dropout_keep=0.9,

    #Bert model parameters
    bert_config_file='./pretrained/bert_chinese/bert_config.json',
    bert_init_checkpoint='./pretrained/bert_chinese/bert_model.ckpt',
    #word2vec pretrained
    word2vec_file='./pretrained/word2vec_cn.cbow.bin',
    word2vec_file2='./pretrained/sgns.merge.word.bz2',
    fasttext_file='./pretrained/fasttext_cc.zh.300.vec.gz',

    #
    len_train_examples=None,
    len_eval_examples=None,
    len_test_examples=None,

    #
    penalization=False,

    #
    dense_hidden_size=512,
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

    #config.to_json_string('./config.json',params)
    config.from_json_file('./config.json')
    print(config.params)


