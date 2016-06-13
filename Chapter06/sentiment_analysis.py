import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
 
if __name__=='__main__':
    # Load positive and negative reviews  
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
     
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 
            'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 
            'Negative') for f in negative_fileids]
     
    # Split the data into train and test (80/20)
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))
     
    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
    print "\nNumber of training datapoints:", len(features_train)
    print "Number of test datapoints:", len(features_test)
     
    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print "\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test)

    print "\nTop 10 most informative words:"
    for item in classifier.most_informative_features()[:10]:
        print item[0]

    # Sample input reviews
    input_reviews = [
        "It is an amazing movie", 
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie", 
        "The direction was terrible and the story was all over the place" 
    ]

    print "\nPredictions:"
    for review in input_reviews:
        print "\nReview:", review
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print "Predicted sentiment:", pred_sentiment 
        print "Probability:", round(probdist.prob(pred_sentiment), 2)

