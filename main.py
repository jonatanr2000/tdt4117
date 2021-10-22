import random
import codecs
import string
import copy
import gensim
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer

random.seed(123)

# Task 1
def load_file():
    # Loads file
    return codecs.open("./2852991.txt", "r", "utf-8")

def get_paragraphs(file):
    # Gets the paragraphs from the file
    return file.read().split('\n\n')

def remove_gutenberg(text):
    # Remove strings containing 'Gutenberg'
    return [x for x in text if 'gutenberg' not in x.lower()]

def split_on_space(text):
    # Split strings on space
    return [x.split(' ') for x in text]

def remove_punctuation(text, paragraphs=True):
    # Create removal-list
    remove = str.maketrans('', '', string.punctuation+"\n\r\t")
    # Remove punctuation and escape characters 
    if paragraphs:
        return [[word.translate(remove) for word in strings] for strings in text]
    return [word.translate(remove) for word in text]

def stem_text(text, paragraphs=True):
    # Inialize stemmer and stem the words
    stemmer = PorterStemmer()
    if paragraphs:
        return [[stemmer.stem(word.lower()) for word in strings] for strings in text]
    return [stemmer.stem(word.lower()) for word in text]

# Task 2

def build_dict(text):
    # Build dictionary
    return gensim.corpora.Dictionary(text)

def get_stopwords():
    # Getting stop words
    f = codecs.open("./common-english-words.txt", "r", "utf-8")
    return f.read().split(',')

def get_stopwords_id(stopwords, dictionary):
    # Getting ids of stopwords based on dictionary
    ids = []
    for word in stopwords:
        try:
            ids.append(dictionary.token2id[word])
        except:
            pass
    return ids

def filter_stopwords(stop_ids, dictionary):
    # Filter stopwords
    return dictionary.filter_tokens(stop_ids)

def bow(stemmed_paragraphs, dictionary):
    # Mapping to BoW
    bag_of_words = []
    for paragraph in stemmed_paragraphs:
        bag_of_words.append(dictionary.doc2bow(paragraph))
    return bag_of_words, dictionary

# Task 3

def build_tfidf_model(bow):
    # Build tfid model
    return gensim.models.TfidfModel(bow)

def build_tfidf_corpus(tfidf_model, bow):
    # Build corpus of tfid
    return tfidf_model[bow]

def generate_sim_matrix(tfidf_corpus):
    # Make similarity matrix
    return gensim.similarities.MatrixSimilarity(tfidf_corpus)

def generate_lsi_model(tfidf_corpus, dictionary, topics=100):
    # Generate LSI model
    return gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=topics)

# Task 4

def preprocessing(text):
    text = text.split(' ')
    text = remove_punctuation(text, paragraphs=False)
    text = stem_text(text, paragraphs=False)
    return text

def show_docs(docs, paragraphs):
    # This function displays documents on a given format.
    for doc in docs:
        p = doc[0]
        print("\n[Paragraph " + p.__str__() + "]")
        print(paragraphs[p][:200])

def show_topics(topics, lsi_model):
    # This function displays topics on a given format.
    for topic in enumerate(topics):
        t = topic[1][0]
        print("\n[Topic " + t.__str__() + "]")
        print(lsi_model.show_topics()[t])


if __name__ == '__main__':
    # 1.1
    f = load_file()
    # 1.2
    paragraphs = get_paragraphs(f)
    # 1.3
    paragraphs = remove_gutenberg(paragraphs)
    paragraphs_unedited = copy.copy(paragraphs)
    # 1.4
    paragraphs = split_on_space(paragraphs)
    paragraphs = remove_punctuation(paragraphs)
    # 1.5
    stemmed_paragraphs = stem_text(paragraphs)

    # 2.1
    dictionary = build_dict(stemmed_paragraphs)
    stopwords = get_stopwords()
    stop_ids = get_stopwords_id(stopwords, dictionary)
    filtered = filter_stopwords(stop_ids, dictionary)
    # 2.2
    bag_of_words, dictionary = bow(stemmed_paragraphs, dictionary)
    
    # 3.1
    tfidf_model = build_tfidf_model(bag_of_words)
    # 3.2 
    tfidf_corpus = build_tfidf_corpus(tfidf_model, bag_of_words)
    # 3.3
    sim_matrix = generate_sim_matrix(tfidf_corpus)
    # 3.4
    lsi_model = generate_lsi_model(tfidf_corpus, dictionary, topics=100)
    lsi_corpus = lsi_model[bag_of_words]
    lsi_matrix = gensim.similarities.MatrixSimilarity(lsi_corpus)
    # 3.5
    #print(lsi_model.show_topics())

    # 4.1
    query = "What is the function of money?".lower()
    query = preprocessing(query)
    query = dictionary.doc2bow(query)

    # 4.2
    index_query_tfidf = tfidf_model[query]

    # 4.3
    docsim = enumerate(sim_matrix[index_query_tfidf])
    docs = sorted(docsim, key=lambda kv: -kv[1])[:3]

    # 4.4
    lsi_query = lsi_model[query]
    topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
    show_topics(topics, lsi_model)
    docsim = enumerate(lsi_matrix[lsi_query])
    docs = sorted(docsim, key=lambda kv: -kv[1])[:3]
    show_docs(docs, paragraphs_unedited)

    

    

    


