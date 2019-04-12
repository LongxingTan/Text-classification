# encoding:utf-8
import os
import csv
import logging
import tensorflow as tf
from model_params import params,config
from models.bert import TextBert,get_assignment_map_from_checkpoint
from models.bi_lstm import Bi_LSTM
from models.lstm import LSTM
from models.cnn import TextCNN
from models.self_attention import Self_attention
from models.gru_attention import GRU_Attention
from models.capsule import Capsule
from models.c_lstm import C_LSTM
from models.rcnn import RCNN
from models.vdcnn import VDCNN
from prepare_inputs import OnlineProcessor,file_based_input_fn_builder,input_fn_builder
from models._loss import create_loss
from models._optimization import create_optimizer_basic_adam,create_optimizer_warmup_adam
from models._eval import create_eval,create_eval_sk
from tokenization import create_vocab_and_label

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


new_data=True  # if it's new data for model
if not new_data:
    # The already used data to select the model or predict, config use the saved json
    config.from_json_file('./config.json')
    params = config.params
    if os.path.exists(os.path.join(params['data_dir'],'train.tf_record')):
        tf.logging.debug("Run from the existing 'train.tf_record', the config will refer to config.json")
    else: tf.logging.debug("Please double check, you want to run new data, with existing train.tf_record?")

else:
    # new data for model
    tf.logging.debug('Run with new data, generate the tf_record/vocab/label, config will refer to the model_params.py')
    create_vocab_and_label(params)


def model_fn_builder(textmodel,params,init_checkpoint=None):
    def model_fn(features, labels, mode):
        inputs, targets = features['input_ids'], features['label_ids']
        model=textmodel(training=(mode==tf.estimator.ModeKeys.TRAIN),params=params)
        logits = model(inputs, targets)
        targets_onehot = tf.one_hot(targets, depth=params['n_class'],dtype=tf.float32)

        loss = create_loss(logits=logits, y_onehot=targets_onehot, loss_type='cross_entropy')
        prediction_label = tf.argmax(logits, -1,output_type=tf.int32)
        probabilities = tf.reduce_max(tf.nn.softmax(logits, name='softmax_tensor'),axis=-1)
        #accuracy= tf.metrics.accuracy(labels=targets, predictions=prediction_label)

        correct_predictions = tf.equal(prediction_label,targets)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        logging_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": accuracy}, every_n_iter=100)
        tf.summary.scalar('accuracy', accuracy)

        if init_checkpoint: # for bert
            tvars = tf.trainable_variables()
            (assignment_map,initialized_variable_names)=get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint,assignment_map)

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("**** Start predict ****")
            prediction_dict = {"labels": prediction_label,
                               'probabilities':probabilities}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=prediction_dict)


        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("**** Start evaluate ****")
            eval_metric_ops = create_eval(targets, prediction_label)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

        else:
            #train_op=create_optimizer_basic_adam(loss,learning_rate=params['learning_rate'])
            num_train_steps = int(params['len_train_examples'] / params['batch_size'] * params['num_train_epochs'])
            train_op=create_optimizer_warmup_adam(loss,init_learning_rate=params['learning_rate'],num_train_steps=num_train_steps,num_warmup_steps=int(0.10*num_train_steps))
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[logging_hook])
    return model_fn


def run_classifier(textmodel,params,data_process_class,init_checkpoint=None):
    model_fn=model_fn_builder(textmodel,params=params,init_checkpoint=init_checkpoint)
    estimator= tf.estimator.Estimator(model_fn=model_fn, model_dir=params["model_dir"])

    if not params['file_based']:
        train_input_fn=input_fn_builder(features=data_process_class.get_train_examples(data_dir=params['data_dir']),
                                        batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=True)
    else:
        train_input_fn=file_based_input_fn_builder(input_file=os.path.join(params["data_dir"],"train.tf_record"),
                                                   params=params,is_training=True)
    num_train_steps = int(params['len_train_examples'] / params['batch_size'] * params['num_train_epochs'])
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


    if not params['file_based']:
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

    if not params['file_based']:
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
    online = OnlineProcessor(params=params, seq_length=params["seq_length"], chinese_seg=params['chinese_seg'])

    if new_data:
        online.get_train_examples(data_dir=params['data_dir'],generate_file=True)
        online.get_dev_examples(data_dir=params['data_dir'], generate_file=True)
        online.get_test_examples(data_dir=params['data_dir'],generate_file=True)
        config.to_json_string('./config.json', params)

    #run_classifier(textmodel=TextBert,params=params,data_process_class=online,init_checkpoint=params['bert_init_checkpoint'])
    run_classifier(textmodel=TextCNN,params=params,data_process_class=online)
    #run_classifier(textmodel=Bi_LSTM,params=params,data_process_class=online)
    #run_classifier(textmodel=GRU_Attention,params=params,data_process_class=online)
    #run_classifier(textmodel=Self_attention,params=params,data_process_class=online)
    #run_classifier(textmodel=RCNN,params=params,data_process_class=online)
    #run_classifier(textmodel=C_LSTM, params=params, data_process_class=online)
    #run_classifier(textmodel=Capsule,params=params,data_process_class=online)
    #run_classifier(textmodel=VDCNN, params=params, data_process_class=online)


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
    print('evaluation:',eval)
