from collections import defaultdict
import json
import copy

params=defaultdict(
    batch_size=128,
    num_train_epochs=10,
    seq_length=50,
    vocab_size=None,
    chinese_seg='char', #word or char
    embedding_size=300,
    kernel_sizes=[2,4,6,14],
    filters=32,
    n_class=None,
    model_dir='./outputs',
    data_dir='./data',
    output_dir='./outputs',
    log_dir='./log',
    learning_rate=10e-3,
    learning_rate_warmup_steps=500,
    optimizer_adam_beta1=0.1,
    optimizer_adam_beta2=0.1,
    optimizer_adam_epsilon=10e-6,

    # RNN model parameters
    lstm_hidden_size=128,
    gru_hidden_size=128,
    attention_hidden_size=64,

    #Bert model parameters
    bert_config_file='./pretrained/chinese/bert_config.json',
    bert_init_checkpoint='./pretrained/chinese/bert_model.ckpt',
    #word2vec pretrained
    word2vec='/pretrained/cn.cbow.bin',

    #
    len_train_examples=None,
    len_eval_examples=None,
    len_test_examples=None,
)


class Model_config(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        config = Model_config(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


