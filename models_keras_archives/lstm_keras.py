from keras.layers import Input,Embedding,GRU,Bidirectional,GlobalMaxPool1D,Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model,load_model
from utils import *

def RNN_model(sentence_length,vocabulary_size,embedding_dim,n_classes,lstm_outdim=80,rnn_type='lstm',embedding_matrix=None):
    inputs=Input(shape=(sentence_length,),dtype='int32')
    if embedding_matrix is None:
        embedding=Embedding(vocabulary_size,embedding_dim)(inputs)
    else:
        embedding_layer = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix],
                                    input_length=sentence_length, trainable=True)
        embedding = embedding_layer(inputs)
    if rnn_type=='gru' or 'GRU':
        rnn=Bidirectional(GRU(80,return_sequences=True))(embedding)
    elif rnn_type=='lstm' or 'LSTM':
        rnn = Bidirectional(LSTM(lstm_outdim, return_sequences=True))(embedding)
    else:
        raise ("invalid rnn_type input, use gru or lstm")
    pool=GlobalMaxPool1D()(rnn)
    dropout1=Dropout(0.5)(pool)
    outputs=Dense(units=n_classes,activation='softmax')(dropout1)
    model=Model(inputs=inputs,outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    model.save('lstm_keras.h5')
    return model

def rnn_keras_train(x_train,y_train,x_test,y_test,sentence_length,vocabulary_size,embedding_dim,n_classes,embedding_matrix):
    model = RNN_model(sentence_length=sentence_length, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
                      n_classes=n_classes,embedding_matrix=embedding_matrix)
    model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1)
    score = model.evaluate(x_test,y_test, batch_size=64, verbose=1)
    print('Loss:', score[0])
    print('Accuracy:', score[1])
    return model

def rnn_keras_predict_new_data(x_new):
    model=load_model('lstm_keras.h5')
    y_new_onehot = model.predict(x_new)
    y_new_index = np.argmax(y_new_onehot, 1)
    return y_new_index

if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('CAC.csv')
    #x_data, y_data = x_data[:50], y_data[:50] ##delete
    #n_classes=len(np.unique(y_data)) ##delete
    x_token = x_data.apply(tokenize_sentence)
    sentence_padded, sentence_length=pad_sentence(x_token)
    x_token_pad, sentence_length = pad_sentence(x_token)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)

    #print(np.array(x_train).shape,np.array(y_train).shape)
    embedding_matrix=word_embed(word2index)
    rnn_keras_train(x_train,y_train, x_test, y_test,sentence_length=sentence_length,vocabulary_size=len(word2index),
                    embedding_dim=300,n_classes=n_classes,embedding_matrix=embedding_matrix)
    y_new_index = rnn_keras_predict_new_data(x_test)
    y_new_label = [index2label[i] for i in y_new_index]
    y_test_index = np.argmax(y_test, 1)
    y_test_label = [index2label[i] for i in y_test_index]
    x_test_label = [[index2word[i] for i in sentence] for sentence in x_test]

    with open('rnn_keras_result.csv', 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file, delimiter=';')
        w.writerow(('Description', 'Label', 'Label_predicted'))
        w.writerows(zip(x_test_label, y_test_label, y_new_label))