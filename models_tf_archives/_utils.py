#author: Tan longxing
#Nice overview post: http://ruder.io/deep-learning-nlp-best-practices/index.html
import pandas as pd
import numpy as np
import logging
import jieba
import markovify
import itertools
from collections import Counter
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import gensim
import re
import csv

logging.basicConfig(level=logging.DEBUG)
def import_data(file):
    '''import the data, and return the sentence and label for classification'''
    raw_data=pd.read_csv(open(file,'rU',encoding='utf-8'),sep=';',error_bad_lines=False)
    raw_data.columns=['Description','Label1','Label2']
    raw_data.dropna(subset=['Description','Label1'],inplace=True)
    raw_data['Description']=raw_data['Description'].apply(lambda x: x.split('首次')[-1])
    x_data, y_data=raw_data['Description'],raw_data['Label1'].str.strip()

    sample_size=len(x_data)
    n_classes=len(np.unique(y_data))
    logging.info("Samples size: %s" %sample_size)
    logging.info("Label category: %s"%n_classes)
    return x_data, y_data,n_classes

def augment_data(x_data, y_data,augment_size=50):
    '''augment the data by markovify to improve the imbalance issue
    https://towardsdatascience.com/nlg-for-fun-automated-headlines-generator-6d0459f9588f'''
    #To-do, solve the generate None bug
    x_dataaug,y_dataaug=[],[]
    for label in set(y_data):
        label_indices=np.where(y_data==label)
        n_samples=len(label_indices[0])
        if n_samples<augment_size:
            print(label,x_data[label_indices[0]])
            text_model=markovify.NewlineText(x_data[label_indices[0]])
            for i in range(augment_size-n_samples):
                print(text_model.make_sentence(tries=100))
                x_dataaug.append(text_model.make_sentence(tries=100))
                y_dataaug.append(label)
    return np.array(x_dataaug),np.array(y_dataaug)

def tokenize_text(text):
    '''For HAN model, split each text into sentence first,
    notice the Chinese decode probelm,decode('gbk') '''
    cutLineFlag = '"？", "！", "。","，","："'
    line,x_sent=[],[]
    for word in text:
        if word in cutLineFlag:
            x_sent.append(["".join(line)])
            line=[]
        else:
            line.append(word)
    return x_sent

def char_text(text):
    '''process the text to char level for char based models like char CNN'''
    char_text=[[word for word in sentence.split()] for sentence in text]

def tokenize_sentence(sentence,sep=' '):
    '''word segmentation for Chinese based on jieba. Also refer to SnowNLP,pynlpir,thulac'''
    jieba.load_userdict('../data/user_dict.txt')
    stopwords=[line.strip() for line in open('../data/stop_words.txt','r',encoding='utf-8').readlines()]
    seglist=jieba.cut(sentence)
    wordlist='' if sep==' ' else []
    for seg in seglist:
        if seg not in stopwords:
            if sep==' ':
                wordlist+=seg+' '
            else:
                wordlist.append(seg)
        else:
            continue
    return wordlist

def pad_sentence(x_data,sentence_length):
    '''pad the sentence to make sure all sentences have the same shape as input'''
    if sentence_length==None:
        sentence_length = max(len(x) for x in x_data)

    sentence_padded=[]
    x_data=[x.split(" ") for x in x_data]
    for sentence in x_data:
        num_padding=sentence_length-len(sentence)
        sentence_padded.append(sentence[:sentence_length]+num_padding*['<pad>'])
    return sentence_padded,sentence_length

def word_index_transform(x_data):
    '''create the vocabulary dictionary to encode or decode the word'''
    x_data=np.array(x_data)
    word_counts=Counter(itertools.chain(*x_data))
    vocabulary_list=[x[0] for x in word_counts.most_common()]
    word2index={x:i for i,x in enumerate(vocabulary_list)}
    index2word={i:x for i,x in enumerate(vocabulary_list)}
    return word2index,index2word

def onehot_sentence(x_data,n_col=None):
    '''transfer the text into one-hot feature representation.
    Warning: not good for real case, may cause memory error'''
    x_data=np.array(x_data)
    if not n_col:
        n_col=np.amax(x_data)+1
    onehot=np.zeros((x_data.shape[0],x_data.shape[1],n_col))
    for i in range(x_data.shape[0]):
        onehot[i,np.arange(x_data.shape[1]),x_data[i]]=1
    return onehot

def countvectorize(x_data):
    '''Bag of words: Vectorize the text by sklearn'''
    #x_data=np.array(x_data)
    countvec = CountVectorizer()
    x_countvec=countvec.fit_transform(x_data)
    logging.info("Count vectorized shape: %s" % x_countvec.shape[0])
    return x_countvec,countvec

def select_feature_chi(x_datavec,label2index,y_index, countvec,k,if_print=False):
    '''the chi square method to select the feature, refer to:
    https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f'''
    #x_datavec, y_index=np.array(x_datavec),np.array(y_index).reshape(-1,1)
    k = min(x_datavec.shape[1], k)
    logging.info("Chi-square feature: %s" %k)
    skb=SelectKBest(chi2,k=k)
    x_datachi=skb.fit_transform(x_datavec,y_index)

    feature_ids=skb.get_support(indices=True)
    feature_names=countvec.get_feature_names()
    vocab_chi={}

    for new_fid,old_fid in enumerate(feature_ids):
        feature_name=feature_names[old_fid]
        vocab_chi[feature_name]=new_fid

    for label,index in sorted(label2index.items()):
        features_chi2 = chi2(x_datavec, [1 if i==index else 0 for i in y_index])
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(countvec.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        if if_print:
            print("# '{}':".format(label))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-3:])))
    return x_datachi,vocab_chi

def tfidf_weight(x_data,vocab,if_save=False):
    '''calculate the tfidf as the feature weight by sklearn'''
    x_data=np.array(x_data)
    tfidf=TfidfVectorizer(vocabulary=vocab)
    x_data_tfidf=tfidf.fit_transform(x_data)

    feature_names = tfidf.get_feature_names()
    feature, weight = [], []
    for i in range(x_data_tfidf.shape[0]):
        feature_index = x_data_tfidf[i, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [x_data_tfidf[i, x] for x in feature_index])
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            feature.append([w, s])
    if if_save == True:
        with open('word_tfidf.csv', 'w', encoding='utf_8_sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(feature)
    return x_data_tfidf,tfidf,feature

def word_embed_trans(vocabulary,embedding_type,embedding_size=300):
    '''pretrained embedding matrix (trained by Wikipedia Chinese) for fine tuning
    Use different pre-trained word2vec is also a way to improve generalization'''
    if embedding_type=='word2vec':
        embedding_file = '../cn.cbow.bin'
    elif embedding_type=='glove':
        embedding_file='../glove.cn.bin'
    elif embedding_type=='fasttext':
        embedding_file='../fasttext.bin'
    else:
        assert 'only word2vec glove or fasttext is allowed'

    embedding_vocab=gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True, encoding='utf-8',
                                                            unicode_errors='ignore')
    embedding_matrix = np.zeros((len(vocabulary), embedding_size))
    for word, i in vocabulary.items():
        #print(word,i) #排气管 93
        if word in embedding_vocab.vocab:
            embedding_matrix[i] = embedding_vocab[word]
        elif re.search('波箱',word):
            word='变速箱'
            embedding_matrix[i]=embedding_vocab[word]
        else:
            embedding_matrix[i] = np.random.random(embedding_size)
    return embedding_matrix

def golve_embed(vocabulary, glove_file,embedding_size):
    #word2vec pre-trained model Chinese: https://github.com/Embedding/Chinese-Word-Vectors
    #glove pre-trained model Chinese: https://github.com/to-shimo/chinese-word2vec
    #fasttext pre=trained model Chinese: https://github.com/Babylonpartners/fastText_multilingual
    #BERT CHinese:https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    glove_vocab, glove_matrix = {}, np.zeros((len(vocabulary), embedding_size))
    glove_weights = open(glove_file, 'r')
    for line in glove_weights:
        word = line.split()[0]
        embedding = np.array([float(val) for val in line.split()[1:]])
        glove_vocab[word] = embedding
    for word,i in vocabulary.items():
        if word in glove_vocab.keys():
            glove_matrix[i]=glove_matrix[word]
        else:
            glove_matrix[i]=np.random.random(embedding_size)
    return glove_matrix


def elmo_embed():
    pass

def use_tfidf_feature(vocabulary,tfidf_feature):
    tfidf_matrix=np.zeros((len(vocabulary),len(tfidf_feature)))
    tfidf=list(tfidf_feature.keys())
    for word,i in vocabulary.items():
        if word in tfidf_feature.keys():
            tfidf_matrix[i,tfidf.index(word)]=tfidf_feature[word]
    return tfidf_matrix


def create_batch(data,batch_size,n_epochs=1,shuffle=True):
    '''Generate the batch iterator for the dataset'''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(n_epochs):
        if shuffle: # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def create_data(sentence_length,sample=None):
    x_data, y_data, n_classes = import_data('../data_raw/CAC.csv')
    if sample!=None:
        x_data, y_data = x_data[:sample], y_data[:sample]
        n_classes = len(np.unique(y_data))
    x_token = x_data.apply(tokenize_sentence)
    x_token_pad, sentence_length = pad_sentence(x_token, sentence_length=sentence_length)
    word2index, index2word = word_index_transform(x_token_pad)
    logging.info("vocabulary size: %s" % len(word2index))
    x_token_index = [[word2index[i] for i in sentence] for sentence in x_token_pad]
    label2index, index2label = word_index_transform(y_data[:, np.newaxis])
    y_index = [label2index[i] for i in y_data]
    y_onehot = label_binarize(y_index, np.arange(n_classes))
    x_train, x_test, y_train, y_test = train_test_split(x_token_index, y_onehot, test_size=0.1, random_state=0)
    return x_train, x_test, y_train, y_test,word2index,n_classes,index2word,index2label

def weighted_prc():
    '''for imbalance classification, I use prc to evaluate the model
    for too many classes, I adjust its weight'''
    pass

def top_k_categorical_f1(y_true,y_pred,k):
    '''The top k labels from the training labels F1 score is used to evaluate the model performance'''
    top_label=top_k(y_true)
    index=[]
    for i, value in enumerate(y_true):
        if value in top_label:
            index.append(i)
    for j,value in enumerate(y_pred):
        if value in top_label:
            index.append(j)
    index=np.unique(index)
    y_true_top=y_true[index]
    y_pred_top=y_pred[index]


def top_k(y_data):
    y_top = Counter(y_data)
    y_topk = y_top.most_common(50)
    top_label = [item[0] for item in y_topk]
    top_label_quan = sum([item[1] for item in y_topk])
    y_quan = len(y_data)
    print(top_label)
    quantile = top_label_quan / y_quan
    logging.info("Label category: %s" %quantile)
    return top_label

def epoch_plot():
    pass

def roc_plot():
    y_test_onehot = label_binarize(y_test, np.arange(len(set(y_train))))
    fpr, tpr, thresholds = sk.metrics.roc_curve(y_test_onehot.ravel(), y_new_prob.ravel())
    ax = plt.subplot(2, 1, 1)
    ax.step(fpr, tpr, color='b', alpha=0.2, where='post')
    ax.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    ax.set_title('ROC curve')

    precision, recall, th = sk.metrics.precision_recall_curve(y_test_onehot.ravel(), y_new_prob.ravel())
    ax = plt.subplot(2, 1, 2)
    ax.plot(recall, precision)
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision recall curve')
    plt.show()

def isEN(uchar):
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def isZH(char):
    if not ('\u4e00' <= char <= '\u9fa5'):
        return False
    return True
# np.argsort(-testpreds)[:, 0:3]
