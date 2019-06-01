# encoding:utf-8
import os
import csv
import logging
import tensorflow as tf
from config import params,Config
from prepare_inputs import OnlineProcessor,file_based_input_fn_builder,input_fn_builder
from models._eval import create_eval,create_eval_sk
from tokenization import create_vocab_and_label
from prepare_models import model_fn_builder

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_classifier(textmodel,params,data_process_class,init_checkpoint=None):
    model_fn=model_fn_builder(textmodel,params=params,init_checkpoint=init_checkpoint)
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=300, keep_checkpoint_max=5,log_step_count_steps=50)
    estimator= tf.estimator.Estimator(model_fn=model_fn, model_dir=params["model_dir"],config=run_config)

    if not params['use_tf_record']:
        train_input_fn=input_fn_builder(features=data_process_class.get_train_examples(data_dir=params['data_dir']),
                                        batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=True)
    else:
        train_input_fn=file_based_input_fn_builder(input_file=os.path.join(params["data_dir"],"train.tf_record"),
                                                   params=params,is_training=True)
    num_train_steps = int(params['len_train_examples'] / params['batch_size'] * params['num_train_epochs'])
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if not params['use_tf_record']:
        eval_input_fn=input_fn_builder(features=data_process_class.get_dev_examples(data_dir=params['data_dir']),
                                       batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=False)
    else:
        eval_input_fn=file_based_input_fn_builder(input_file=os.path.join(params['data_dir'],'eval.tf_record'),
                                                  params=params,is_training=False)
    eval_steps = int(params['len_dev_examples'] // params['batch_size']+1)
    result=estimator.evaluate(input_fn=eval_input_fn,steps=eval_steps)

    output_eval_file=os.path.join(params['output_dir'], 'eval_result.txt')
    with tf.gfile.GFile(output_eval_file,'w') as writer:
        for key in sorted(result.keys()):
            tf.logging.info(" %s = %s",key,str(result[key]))
            writer.write(" %s = %s\n"%(key,str(result[key])))

    if not params['use_tf_record']:
        predict_input_fn=input_fn_builder(features=data_process_class.get_test_examples(data_dir=params['data_dir']),
                                          batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=False)
    else:
        predict_input_fn=file_based_input_fn_builder(input_file=os.path.join(params['data_dir'],'test.tf_record'),
                                                     params=params,is_training=False)
    result=estimator.predict(input_fn=predict_input_fn)

    predict_file=os.path.join(params['output_dir'], 'test_result.csv')
    with open(predict_file, 'w') as writer:
        for i,prediction in enumerate(result):
            label=prediction['labels']
            probability=prediction['probabilities']
            output_line=",".join([str(label),str(probability)])+'\n'
            writer.write(output_line)


if __name__ == "__main__":
    config = Config()
    if not params['new_data']:
        # The already used data to select the model or predict, config use the saved json
        config=Config()
        config.from_json_file('./config.json')
        params = config.params
        if os.path.exists(os.path.join(params['data_dir'], 'train.tf_record')):
            tf.logging.debug("Run from the existing 'train.tf_record', the config will refer to config.json")
        else:
            tf.logging.debug("Please double check, you want to run new data, with existing train.tf_record?")
    else:
        # new data for model
        tf.logging.debug('Run with new data, generate the tf_record/vocab/label, config will refer to the config.py')
        create_vocab_and_label(params)

    online = OnlineProcessor(params=params, seq_length=params["seq_length"], chinese_seg=params['chinese_seg'])

    if params['new_data']:
        online.get_train_examples(data_dir=params['data_dir'],generate_file=True)
        online.get_dev_examples(data_dir=params['data_dir'], generate_file=True)
        online.get_test_examples(data_dir=params['data_dir'],generate_file=True)
        config.to_json_string('./config.json', params)

    if params['model']=='TextBert':
        run_classifier(textmodel=params['model'],params=params,data_process_class=online,init_checkpoint=params['bert_init_checkpoint'])
    else:
        run_classifier(textmodel=params['model'],params=params,data_process_class=online)

    label, predict = [], []
    index2label, label2index=online.load_label_dict()

    with open('out.csv', 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        result = []
        for line in csv.reader(open('./outputs/test_result.csv', 'r')):
            result.append(line)

        for i, row_test in enumerate(csv.reader(open('./data/test.csv', 'r', encoding="utf-8"))):
            row = []
            row.append(row_test[0])
            row.append(row_test[1])

            if row_test[1] in label2index.keys():
                row.append(label2index[row_test[1]])
                label.append(label2index[row_test[1]])
            else:
                row.append(1000)
                label.append(1000)

            row.append(index2label[int(result[i][0])])
            row.append(result[i][0])
            predict.append(int(result[i][0]))
            row.append(result[i][1])
            writer.writerow(row)

    eval=create_eval_sk(labels=label,predictions=predict)
    print('Evaluation:',eval)
