from nltk.tokenize import RegexpTokenizer  
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
from nltk.corpus import stopwords

# Load input data
def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])

    return data

# Class to preprocess text
class Preprocessor(object):
    # Initialize various operators
    def __init__(self):
        # Create a regular expression tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # get the list of stop words 
        self.stop_words_english = stopwords.words('english')

        # Create a Snowball stemmer 
        self.stemmer = SnowballStemmer('english')
        
    # Tokenizing, stop word removal, and stemming
    def process(self, input_text):
        # Tokenize the string
        tokens = self.tokenizer.tokenize(input_text.lower())

        # Remove the stop words 
        tokens_stopwords = [x for x in tokens if not x in self.stop_words_english]
        
        # Perform stemming on the tokens 
        tokens_stemmed = [self.stemmer.stem(x) for x in tokens_stopwords]

        return tokens_stemmed
    
if __name__=='__main__':
    # File containing linewise input data 
    input_file = 'data_topic_modeling.txt'

    # Load data
    data = load_data(input_file)

    # Create a preprocessor object
    preprocessor = Preprocessor()

    # Create a list for processed documents
    processed_tokens = [preprocessor.process(x) for x in data]

    # Create a dictionary based on the tokenized documents
    dict_tokens = corpora.Dictionary(processed_tokens)
        
    # Create a document-term matrix
    corpus = [dict_tokens.doc2bow(text) for text in processed_tokens]

    # Generate the LDA model based on the corpus we just created
    num_topics = 2
    num_words = 4
    ldamodel = models.ldamodel.LdaModel(corpus, 
            num_topics=num_topics, id2word=dict_tokens, passes=25)

    print "\nMost contributing words to the topics:"
    for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
        print "\nTopic", item[0], "==>", item[1]

