import random
import codecs
import string
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

def remove_punctuation(text):
    # Create removal-list
    remove = str.maketrans('', '', string.punctuation+"\n\r\t")
    # Remove punctuation and escape characters 
    return [[word.translate(remove) for word in strings] for strings in text]

def stem_text(text):
    # Inialize stemmer and stem the words
    stemmer = PorterStemmer()
    return [[stemmer.stem(word.lower()) for word in strings] for strings in text]

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

def build_tfid_model(dictionary):
    # Build tfid model
    return gensim.models.TfidModel(dictionary)

def build_tfid_corpus(tfid_model, bow):
    # Build corpus of tfid
    return tfid_model[bow]

def generate_sim_matrix(tfid_corpus):
    # Make similarity matrix
    return gensim.similarities.MatrixSimilarity(tfid_corpus)

def generate_lsi_model(tfid_corpus, dictionary, topics=100):
    # Generate LSI model
    return gensim.models.LsiModel(tfid_corpus, id2word=dictionary, num_topics=topics)


if __name__ == '__main__':
    # 1.1
    f = load_file()
    # 1.2
    paragraphs = get_paragraphs(f)
    # 1.3
    paragraphs = remove_gutenberg(paragraphs)
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
    tfid_model = build_tfid_model(dictionary)
    # 3.2 
    tfid_corpus = build_tfid_corpus(tfid_model, bag_of_words)
    # 3.3
    sim_matrix = generate_sim_matrix(tfid_corpus)
    # 3.4
    lsi_model = generate_lsi_model(tfid_corpus, dictionary, topics=100)
    lsi_corpus = lsi_model[bag_of_words]
    lsi_matrix = gensim.similarities.MatrixSimilarity(lsi_corpus)
    # 3.5
    lsi_model.show_topics() 

