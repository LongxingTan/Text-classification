
from keras.layers import Input,Dense,Embedding,Conv2D,MaxPool2D
from keras.layers import Reshape,Flatten,Dropout,Concatenate,Lambda
from keras.models import Model,model_from_json
import keras.backend as K
from utils import *
import keras
import matplotlib.pyplot as plt
import os
from Convolutional.config import *

def CNN_model(sentence_length, vocabulary_size, embedding_dim,n_classes,filter_sizes,filter_num,embedding_matrix=None):
    inputs=Input(shape=(sentence_length,),dtype='int32')
    if embedding_matrix is None:
        embedding = Embedding(vocabulary_size, embedding_dim)(inputs)
    else:
        embedding_layer = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix],
                                    input_length=sentence_length, trainable=True)
        embedding = embedding_layer(inputs)

    reshape=Reshape((sentence_length,embedding_dim,1))(embedding)

    print(reshape.shape)

    conv1=Conv2D(filters=filter_num,kernel_size=(filter_sizes[0],embedding_dim),padding='valid',activation='relu')(reshape)
    drop1=Dropout(0.5)(conv1)
    pool1=MaxPool2D(pool_size=(2,1))(drop1)

    conv2=Conv2D(filters=filter_num,kernel_size=(filter_sizes[1],embedding_dim),padding='valid',activation='relu')(reshape)
    drop2=Dropout(0.5)(conv2)
    pool2=MaxPool2D(pool_size=(2,1))(drop2)

    conv3=Conv2D(filters=filter_num,kernel_size=(filter_sizes[2],embedding_dim),padding='valid',activation='relu')(reshape)
    drop3=Dropout(0.5)(conv3)
    pool3=MaxPool2D(pool_size=(2,1))(drop3)

    merged=Concatenate(axis=1)([pool1,pool2,pool3])
    reshape2 = Lambda(lambda x: K.squeeze(x, 2))(merged)
    merged3=Reshape((-1,filter_num,1))(reshape2)

    conv4=Conv2D(filters=filter_num,kernel_size=(filter_sizes[0],filter_num),padding='valid',activation='relu')(merged3)
    drop4=Dropout(0.5)(conv4)
    pool4=MaxPool2D(pool_size=(2,1))(drop4)

    conv5=Conv2D(filters=filter_num,kernel_size=(filter_sizes[1],filter_num),padding='valid',activation='relu')(merged3)
    drop5=Dropout(0.5)(conv5)
    pool5=MaxPool2D(pool_size=(2,1))(drop5)

    conv6=Conv2D(filters=filter_num,kernel_size=(filter_sizes[2],filter_num),padding='valid',activation='relu')(merged3)
    drop6=Dropout(0.5)(conv6)
    pool6=MaxPool2D(pool_size=(2,1))(drop6)

    merged2=Concatenate(axis=1)([pool4,pool5,pool6])

    flatten=Flatten()(merged2)
    #dropout
    outputs=Dense(units=n_classes,activation='softmax')(flatten)
    #outputs=Dense(200,activation='sigmoid')(dense1)

    model=Model(inputs=inputs,outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    #plot_model(model,show_shapes=True,to_file='multichannel.png')
    #model.save('CAC_keras_cnn.h5')
    cnn_keras = model.to_json()
    with open("cnn_keras_deep.json", "w") as json_file:
        json_file.write(cnn_keras)
    return model

def CNN_keras_train(x_train,y_train,x_test,y_test,sentence_length,vocabulary_size,embedding_dim,n_classes,embedding_matrix):
    model = CNN_model(sentence_length=sentence_length, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
                      n_classes=n_classes,filter_num=256,filter_sizes=[2,4,6],embedding_matrix=embedding_matrix)
    check = keras.callbacks.ModelCheckpoint('cac_keras_deep.hdf5',monitor='val_acc', verbose=1,
                                            save_best_only=True, save_weights_only=False, mode='auto', period=1)
    deepcnn=model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1,validation_data=(x_test,y_test),callbacks=[check])
    score = model.evaluate(x_test,y_test, batch_size=64, verbose=1)
    print('Loss:', score[0])
    print('Accuracy:', score[1])

    fig = plt.figure()
    plt.plot(deepcnn.history['loss'], 'r', linewidth=3.0)
    plt.plot(deepcnn.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves :CAC', fontsize=16)
    fig.savefig('loss_deepcnn.png')
    plt.show()
    return model

def CNN_keras_predict_new_data(x_new):
    if os.path.exists('cnn_keras_deep.json'):
        json_file = open('cnn_keras_deep.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    else:
        model=CNN_keras_train(x_train, y_train, x_test, y_test, sentence_length=sentence_length,
                              vocabulary_size=len(word2index),embedding_dim=300, n_classes=n_classes, embedding_matrix=None)
    y_new_onehot=model.predict(x_new)
    y_new_index=np.argmax(y_new_onehot,1)
    return y_new_index


if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('../CAC.csv')
    #x_data, y_data = x_data[:50], y_data[:50] ##delete
    #n_classes=len(np.unique(y_data)) ##delete
    x_token = x_data.apply(tokenize_sentence)
    x_token_pad, sentence_length = pad_sentence(x_token,sentence_length=config.sentence_length)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    embedding_matrix=word_embed(word2index,'word2vec')
    CNN_keras_train(x_train,y_train, x_test, y_test,sentence_length=sentence_length,vocabulary_size=len(word2index),
                    embedding_dim=300,n_classes=n_classes,embedding_matrix=embedding_matrix)
    y_new_index=CNN_keras_predict_new_data(x_test)
    y_new_label = [index2label[i] for i in y_new_index]
    y_test_index=np.argmax(y_test,1)
    y_test_label=[index2label[i] for i in y_test_index]
    x_test_label=[[index2word[i] for i in sentence] for sentence in x_test]

    with open('cnn_keras_deep_result.csv','w',newline='',encoding='utf-8-sig') as file:
        w=csv.writer(file,delimiter=';')
        w.writerow(('Description','Label','Label_predicted'))
        w.writerows(zip(x_test_label,y_test_label,y_new_label))