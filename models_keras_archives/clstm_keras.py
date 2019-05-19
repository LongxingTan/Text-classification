
from keras.layers import Input,Embedding,Dense,Conv2D,Dropout,Lambda
from keras.layers.recurrent import LSTM
from keras.models import Model,model_from_json
from keras.layers.core import Reshape
import keras.backend as K
from utils import *
import os
from Recurrent.config import *

def CLSTM_model(sentence_length, vocabulary_size, embedding_dim,n_classes,filter_sizes,filter_num,lstm_dim,embedding_matrix=None):
    inputs=Input(shape=(sentence_length,),dtype='int32')
    if embedding_matrix is None:
        embedding = Embedding(vocabulary_size, embedding_dim)(inputs)
    else:
        embedding_layer = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix],
                                    input_length=sentence_length, trainable=True)
        embedding = embedding_layer(inputs)

    reshape=Reshape((sentence_length,embedding_dim,1))(embedding)
    conv_output=Conv2D(filters=filter_num,kernel_size=(filter_sizes[0],embedding_dim),padding='valid',activation='relu',strides=1)(reshape)

    conv_output2=Lambda(lambda x: K.squeeze(x,2))(conv_output)
    lstm_output = LSTM(lstm_dim)(conv_output2)
    dropout_output=Dropout(0.5)(lstm_output)
    outputs = Dense(n_classes,activation="softmax")(dropout_output)
    model=Model(inputs=inputs,outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    clstm_keras = model.to_json()
    with open("clstm_keras_deep.json", "w") as json_file:
        json_file.write(clstm_keras)
    return model

def clstm_keras_train(x_train,y_train,x_test,y_test,sentence_length,vocabulary_size,embedding_dim,n_classes,
                      embedding_matrix,filter_sizes,filter_num,lstm_dim):
    model = CLSTM_model(sentence_length=sentence_length, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
                        n_classes=n_classes,embedding_matrix=embedding_matrix,filter_sizes=filter_sizes,
                        filter_num=filter_num,lstm_dim=lstm_dim)
    model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.n_epochs, verbose=1)
    score = model.evaluate(x_test,y_test, batch_size=64, verbose=1)
    print('Loss:', score[0])
    print('Accuracy:', score[1])
    return model

def clstm_keras_predict_new_data(x_new):
    if os.path.exists('clstm_keras_deep.json'):
        json_file = open('clstm_keras_deep.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    else:
        model=clstm_keras_train(x_train, y_train, x_test, y_test, sentence_length=sentence_length,
                              vocabulary_size=len(word2index),embedding_dim=300, n_classes=n_classes, embedding_matrix=False)
    y_new_onehot=model.predict(x_new)
    y_new_index=np.argmax(y_new_onehot,1)
    return y_new_index

if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('../CAC.csv')
    #x_data, y_data = x_data[:100], y_data[:100] ##delete
    #n_classes=len(np.unique(y_data)) ##delete
    x_token = x_data.apply(tokenize_sentence)
    x_token_pad, sentence_length = pad_sentence(x_token,config.sentence_length)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #print(np.array(x_train).shape,np.array(y_train).shape)
    embedding_matrix=word_embed(word2index,'word2vec')
    clstm_keras_train(x_train,y_train, x_test, y_test,sentence_length=sentence_length,vocabulary_size=len(word2index),
                      embedding_dim=300,n_classes=n_classes,embedding_matrix=embedding_matrix,filter_sizes=[4],
                      filter_num=128,lstm_dim=256)
    y_new_index = clstm_keras_predict_new_data(x_test)
    y_new_label = [index2label[i] for i in y_new_index]
    y_test_index = np.argmax(y_test, 1)
    y_test_label = [index2label[i] for i in y_test_index]
    x_test_label = [[index2word[i] for i in sentence] for sentence in x_test]

    with open('clstm_keras_deep_result.csv', 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file, delimiter=';')
        w.writerow(('Description', 'Label', 'Label_predicted'))
        w.writerows(zip(x_test_label, y_test_label, y_new_label))

#paper reference:
#arxiv.org/abs/1511.08630