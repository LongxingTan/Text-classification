import os
import csv
import logging
import tensorflow as tf
from models._model_params import params
from models.cnn import TextCNN
from models.bert import TextBert,get_assignment_map_from_checkpoint
from models.bi_lstm import Bi_LSTM
from prepare_inputs import OnlineProcessor,file_based_input_fn_builder,input_fn_builder
from models._loss import create_loss
from models._optimization import create_optimizer_basic_adam,create_optimizer_warmup_adam
from models._eval import create_eval


# without logging configuration, tensorflow won't print the training information
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def prepare_input():
    online=OnlineProcessor(seq_length=params["seq_length"],chinese_seg=params['chinese_seg'])
    train_features=online.get_train_examples(data_dir=params['data_dir'])
    dev_features=online.get_dev_axamples(data_dir=params['data_dir'])
    test_features=online.get_test_examples(data_dir=params['data_dir'])
    return train_features,dev_features,test_features



def model_fn_builder(textmodel,init_checkpoint=None):
    def model_fn(features, labels, mode):
        inputs, targets = features['input_ids'], features["label_ids"]


        model=textmodel(training=(mode==tf.estimator.ModeKeys.TRAIN))

        logits = model(inputs, targets)

        targets_onehot = tf.one_hot(targets, depth=params['n_class'],dtype=tf.float32)

        loss = create_loss(logits=logits, y_onehot=targets_onehot, loss_type='cross_entropy')
        prediction_label = tf.argmax(logits, -1,output_type=tf.int32)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')



        if init_checkpoint:
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            (assignment_map,initialized_variable_names)=get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint,assignment_map)
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)



        if mode == tf.estimator.ModeKeys.PREDICT:
            print("Start predict ****")
            prediction_dict = {"labels": prediction_label,
                           'probabilities':probabilities}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=prediction_dict)


        elif mode == tf.estimator.ModeKeys.EVAL:
            print("Start evaluate ****")

            eval_metric_ops = create_eval(targets, prediction_label)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

        else:
            train_op=create_optimizer_basic_adam(loss,learning_rate=params['learning_rate'])
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)
    return model_fn


def run_classifier(textmodel,init_checkpoint=None,train_features=None,dev_features=None,test_features=None):
    model_fn=model_fn_builder(textmodel,init_checkpoint=init_checkpoint)
    estimator= tf.estimator.Estimator(model_fn=model_fn, model_dir=params["model_dir"])

    num_train_steps=int(params['len_train_examples']/params['batch_size']*params['num_train_epochs'])
    if train_features:
        train_input_fn=input_fn_builder(train_features,labels=None,batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=True)
    else:
        train_input_fn=file_based_input_fn_builder(input_file=os.path.join(params["data_dir"],"train.tf_record"), params=params,is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    eval_steps=int(params['len_dev_examples']//params['batch_size'])
    if dev_features:
        eval_input_fn=input_fn_builder(dev_features,labels=None,batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=False)
    else:
        eval_input_fn=file_based_input_fn_builder(input_file=os.path.join(params['data_dir'],'eval.tf_record'),
                                              params=params,
                                              is_training=False)
    result=estimator.evaluate(input_fn=eval_input_fn,steps=eval_steps)

    output_eval_file=os.path.join(params['output_dir'], '1eval_result.txt')
    with tf.gfile.GFile(output_eval_file,'w') as writer:
        for key in sorted(result.keys()):
            tf.logging.info(" %s = %s",key,str(result[key]))
            writer.write(" %s = %s\n"%(key,str(result[key])))

    if test_features:
        predict_input_fn=input_fn_builder(test_features,labels=None,batch_size=params['batch_size'],seq_length=params['seq_length'],is_training=False)
    else:
        predict_input_fn=file_based_input_fn_builder(input_file=os.path.join(params['data_dir'],'test.tf_record'),
                                                 params=params,
                                                 is_training=False)
    result=estimator.predict(input_fn=predict_input_fn)

    predict_file=os.path.join(params['output_dir'], 'test_result.csv')
    with tf.gfile.GFile(predict_file, 'w') as writer:
        for i,prediction in enumerate(result):

            label=prediction['labels']
            output_line=",".join(['text',str(label)])+'\n'
            writer.write(output_line)


if __name__=="__main__":
    train_features, dev_features, test_features=prepare_input()
    #run_classifier(textmodel=TextBert,init_checkpoint=params['bert_init_checkpoint'])
    #run_classifier(textmodel=TextCNN,init_checkpoint=None,train_features=train_features,dev_features=dev_features,test_features=test_features)
    run_classifier(textmodel=Bi_LSTM,init_checkpoint=None,train_features=train_features,dev_features=dev_features,test_features=test_features)

    label_dict = {}
    reader = csv.reader(open('label_dict.csv', 'r'))

    for row in reader:
        label_dict.update({row[1]: row[0]})

    with open('./outputs/test_result.csv', 'r') as f:
        reader = csv.reader(f)
        with open('out.csv', 'w', newline='') as ouf:
            writer = csv.writer(ouf)

            for row in reader:
                row[0] = label_dict[row[1]]
                writer.writerow(row)





