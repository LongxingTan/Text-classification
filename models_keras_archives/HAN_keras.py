#paper reference:
##http://www.aclweb.org/anthology/N16-1174
#code reference:
##https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py

from keras.layers import Input,Embedding,GRU,Bidirectional,Dense,Dropout,TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model,load_model
from keras import backend as K
from keras import initializers
from utils import *
import tensorflow as tf
import tensorflow.contrib.layers as layers
import keras
from keras.engine.topology import Layer

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class HAN():
    def __init__(self,x_input,y_input,vocabulary_size,embedding_dim,embedding_matrix,each_sentence_length,n_sentences,lstm_outdim,attention_dim):
        self.x_input=x_input
        self.y_input=y_input
        self.vocabulary_size=vocabulary_size
        self.embedding_dim=embedding_dim
        self.embedding_matrix=embedding_matrix
        self.each_sentence_length=each_sentence_length
        self.n_sentences=n_sentences
        self.lstm_outdim=lstm_outdim
        self.attention_dim=attention_dim

    def model(self):
        sentence_inputs=Input(shape=(self.each_sentence_length,),name='sentence_input',dtype='int32')
        if self.embedding_matrix is None:
            embedding = Embedding(self.vocabulary_size, self.embedding_dim)(sentence_inputs)
        else:
            embedding_layer = Embedding(self.vocabulary_size, self.embedding_dim, weights=[self.embedding_matrix],
                                        input_length=self.each_sentence_length, trainable=True)
            embedding = embedding_layer(sentence_inputs)

        sentence_lstm=Bidirectional(GRU(self.lstm_outdim,return_sequences=True))(embedding)
        sentence_encoder=AttLayer(100)(sentence_lstm)
        sent_encoder=Model(sentence_inputs,sentence_encoder)

        text_inputs=Input(shape=(self.n_sentences,self.each_sentence_length),dtype='int32')
        text_encoder=TimeDistributed(sent_encoder)(text_inputs)
        text_lstm=Bidirectional(GRU(self.lstm_outdim,return_sequences=True))(text_encoder)
        text_att=AttLayer(100)(text_lstm)
        preds=Dense(n_classes,activation='softmax')(text_att)

        self.model=Model(text_inputs,preds)
        return self.model

    def embedding_layer(self):
        pass

    def bi_gru_encoder(self,inputs,lstm_units,):
        pass

    def _attention_(self,inputs):
        W=K.random_normal_variable(shape=(inputs.shape[-1],self.attention_dim), mean=0, scale=1)
        b=K.random_normal_variable(shape=(self.attention_dim,), mean=0, scale=1)
        u=K.random_normal_variable(shape=(self.attention_dim,1), mean=0, scale=1)

        uit=K.tanh(K.bias_add(K.dot(inputs,W),b))
        ait=K.dot(uit,u)
        ait=K.squeeze(ait,-1)
        ait=K.exp(ait)

        ait = K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = inputs * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def attention(self,inputs):
        w=tf.get_variable(shape=[self.attention_dim],dtype=tf.float32,name='attention_vector')
        input_projection=layers.fully_connected(inputs,self.attention_dim,activation_fn=tf.tanh)
        vector_att=tf.reduce_sum(tf.multiply(input_projection,w),axis=2,keep_dims=True)
        print(vector_att.shape)
        attention_weights=tf.nn.softmax(vector_att,dim=1)
        weighted_projection=tf.multiply(inputs,attention_weights)
        outputs=tf.reduce_sum(weighted_projection,axis=1)
        outputs=K.cast(outputs, dtype='float32')
        keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
        return outputs

    def train(self):
        model=self.model()
        print(self.model.summary())
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
        model.fit(self.x_input,self.y_input,epochs=20,batch_size=32,verbose=1)

    def evaluate(self,x_test,y_test):
        score =self.model.evaluate(x_test, y_test, batch_size=64, verbose=1)
        print('Loss:', score[0])
        print('Accuracy:', score[1])


if '__main__' == __name__:
    x_data, y_data, n_classes = import_data('CAC.csv')
    #x_data, y_data = x_data[:50], y_data[:50]  ##delete
    x_sent = x_data.apply(tokenize_text)
    x_token = [[list(map(tokenize_sentence, sentence)) for sentence in text] for text in x_sent]
    x_token_han=[[sentence[0].split(' ') for sentence in text] for text in x_token]
    word2index, index2word = word_index_transform([sentence[0].split(' ') for text in x_token for sentence in text])

    n_sentences=5
    each_sentence_length=15
    x_token_index=np.zeros((len(x_token_han),n_sentences,each_sentence_length),dtype='int32')
    for i,text in enumerate(x_token_han):
        for j,sentence in enumerate(text):
            if j<n_sentences:
                for k,word in enumerate(sentence):
                    if k<each_sentence_length:
                        x_token_index[i,j,k]=word2index[word]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)
    embedding_matrix = word_embedding(word2index)

    han=HAN(x_input=x_train,y_input=y_train,vocabulary_size=len(word2index),embedding_dim=300,embedding_matrix=embedding_matrix,
            each_sentence_length=each_sentence_length,n_sentences=n_sentences,lstm_outdim=128,attention_dim=128)
    han.train()
    han.evaluate(x_test,y_test)
