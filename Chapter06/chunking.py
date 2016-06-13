import numpy as np
from nltk.corpus import brown

# Split a text into chunks 
def splitter(data, num_words):
    words = data.split(' ')
    output = []

    cur_count = 0
    cur_words = []
    for word in words:
        cur_words.append(word)
        cur_count += 1
        if cur_count == num_words:
            output.append(' '.join(cur_words))
            cur_words = []
            cur_count = 0

    output.append(' '.join(cur_words) )

    return output 

if __name__=='__main__':
    # Read the data from the Brown corpus
    data = ' '.join(brown.words()[:10000])

    # Number of words in each chunk 
    num_words = 1700

    chunks = []
    counter = 0

    text_chunks = splitter(data, num_words)

    print "Number of text chunks =", len(text_chunks)
