from prepare_inputs import OnlineProcessor
from model_params import params

def prepare_predict_input():
    online = OnlineProcessor(seq_length=params["seq_length"], chinese_seg=params['chinese_seg'])
    test_features = online.get_test_examples(data_dir=params['data_dir'])
