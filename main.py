import random
import codecs
import string
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer


random.seed(123)

f = codecs.open("./2852991.txt", "r", "utf-8")

# Load file
textfile = f.read().split('\n\n')
# Remove strings containing 'Gutenberg'
textfile = [x for x in textfile if 'gutenberg' not in x.lower()]
# Split strings on space
textfile = [x.split(' ') for x in textfile]

# Create removal-list
remove = str.maketrans('', '', string.punctuation+"\n\r\t")
# Remove punctuation and escape characters 
textfile = [[word.translate(remove) for word in strings] for strings in textfile]

# Inialize stemmer and stem the words
stemmer = PorterStemmer()
textfile = [[stemmer.stem(word.lower()) for word in strings] for strings in textfile]
print(textfile)
#freq = FreqDist()
#print(freq["tax"])

