text = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."

# Sentence tokenization
from nltk.tokenize import sent_tokenize

sent_tokenize_list = sent_tokenize(text)
print "\nSentence tokenizer:"
print sent_tokenize_list

# Create a new word tokenizer
from nltk.tokenize import word_tokenize

print "\nWord tokenizer:"
print word_tokenize(text)

# Create a new punkt word tokenizer
from nltk.tokenize import PunktWordTokenizer

punkt_word_tokenizer = PunktWordTokenizer()
print "\nPunkt word tokenizer:"
print punkt_word_tokenizer.tokenize(text)

# Create a new WordPunct tokenizer
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()
print "\nWord punct tokenizer:"
print word_punct_tokenizer.tokenize(text)

