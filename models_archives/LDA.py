from models_tf_archives._utils import *
from gensim.corpora import Dictionary
from gensim import models

class LDA_gensim:
    def __init__(self,train_set):
        dictionary = Dictionary(train_set)
        dictionary.filter_extremes(no_below=5, no_above=0.1)
        corpus = [dictionary.doc2bow(text) for text in train_set]

        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100,chunksize=2000,iterations=500,alpha='auto')
        lda.print_topics(20)
        lda.save('lda_35')
        topics_matrix = lda.show_topics(formatted=False, num_words=20)
        for i in range(3):
            topics_array = np.array(topics_matrix[i][1])
            print([str(word[0]) for word in topics_array])


x_data, y_data, n_classes = import_data('../CAC.csv')
#x_data, y_data = x_data[:1000], y_data[:1000] ##delete
x_token = x_data.apply(lambda x: tokenize_sentence(sentence=x,sep='[]'))

#print(x_token)
#tfidf=models.TfidfModel(x_token)
#corpus_tfidf=tfidf[x_token]

LDA_gensim(train_set=x_token)
#print(lda)

lda = models.LdaModel.load('lda_35')
topic_dist = lda.state.get_lambda()

# get topic terms
num_words = 15
topic_terms = [{w for (w, _) in lda.show_topic(topic, topn=num_words)} for topic in range(topic_dist.shape[0])]
print(topic_terms)


from scipy.spatial.distance import pdist, squareform
from gensim.matutils import jensen_shannon
import itertools as itt
def distance(X, dist_metric):
    return squareform(pdist(X, lambda u, v: dist_metric(u, v)))

topic_distance = distance(topic_dist, jensen_shannon)

# store edges b/w every topic pair along with their distance
edges = [(i, j, {'weight': topic_distance[i, j]})
         for i, j in itt.combinations(range(topic_dist.shape[0]), 2)]

# keep edges with distance below the threshold value
k = np.percentile(np.array([e[2]['weight'] for e in edges]), 20)
edges = [e for e in edges if e[2]['weight'] < k]
print(edges)
