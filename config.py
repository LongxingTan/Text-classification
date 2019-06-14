from collections import defaultdict
import json


params_important={
    'model':"TextRNN",  # options: TextBert, TextCNN, TextRNN, RCNN, CLSTM, TextCapsule, VDCNN, BiLSTM, GRUAttention, SelfAttention,
    'chinese_seg':'char',  # options: char, word, mix
    'embedding_size':300,
    'embedding_type':'random', # options: random, word2vec_static, word2vec_finetune, fasttext_static, fasttext_finetune, multi_channel
    'embedding_dropout_keep':0.95,
    'kernel_sizes':[3,5,7],
    'filters':128,
    'rnn_hidden_size':512,
    'attention_hidden_size':128,
    'rnn_dropout_keep':0.95,
    'capsule_vec_length':16,
}

params_necessary=defaultdict(
    new_data=True,  # if True: use config.py and generate .tf_record, otherwise use config.json and tf_record
    do_train=True,
    do_eval=True,
    do_predict=True,
    use_rnn_cell='gru',  # options: lstm, gru
    use_birnn=True,
    use_attention=True,
    use_attention_head=1,
    use_pooling=False,
    seq_length=200,  # sentence length
    use_tf_record=True,  # if use binary tf_reocrd to accelerate the training
    batch_size=128,
    num_train_epochs=7,
    learning_rate=1e-4,  # original value if use warm up
    vocab_size=None,
    n_class=None,
    model_dir='./outputs/checkpoint',
    data_dir='./data',
    output_dir='./outputs',
    log_dir='./logs',
    # Bert model parameters
    bert_config_file='./pretrained/bert_chinese/bert_config.json',
    bert_init_checkpoint='./pretrained/bert_chinese/bert_model.ckpt',
    # word2vec pretrained
    word2vec_file='./pretrained/word2vec_cn.cbow.bin',
    word2vec_file2='./pretrained/sgns.merge.word.bz2',
    fasttext_file='./pretrained/fasttext_cc.zh.300.vec.gz',
    #
    len_train_examples=None,
    len_dev_examples=None,
    len_test_examples=None,
    #
    penalization=False,
    dense_hidden_size=1024,
    dropout_keep=0.95,
    l2_scale=0.001,
)


params = {**params_important, **params_necessary}


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,type):
            return repr(obj)
        return json.JSONEncoder.default(self, obj)


class Config(object):
    def __init__(self):
        self.params=defaultdict()

    def from_json_file(self,json_file):
        with open(json_file, 'r') as f:
            self.params = json.load(f)

    def to_json_string(self,json_file,params):
        with open(json_file, 'w') as f:
            json.dump(params, f,cls=JsonEncoder)


if __name__=='__main__':
    #config.to_json_string('./config.json',params)
    #config.from_json_file('./config.json')
    print(params)
    config = Config()
    config.to_json_string('./config.json', params)
