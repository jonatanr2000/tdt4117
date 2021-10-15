import random
import codecs
import string

random.seed(123)

f = codecs.open("./2852991.txt", "r", "utf-8")

textfile = f.read().split('\n\n')
textfile = [x for x in textfile if 'gutenberg' not in x.lower()]
textfile = [x.split(' ') for x in textfile]
#print(textfile)
remove = string.punctuation+"\n\r\t"
for str_lst in textfile:
    for word in str_lst:
        word = word.translate(remove)
print(textfile)
#textfile = [word.translate(None, string.punctuation) for word in [strings for strings in textfile]]
#print(textfile)