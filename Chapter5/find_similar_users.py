import json
import numpy as np

from pearson_score import pearson_score

# Finds a specified number of users who are similar to the input user
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('User ' + user + ' not present in the dataset')

    # Compute Pearson scores for all the users
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if user != x])

    # Sort the scores based on second column
    scores_sorted = np.argsort(scores[:, 1])

    # Sort the scores in decreasing order (highest score first) 
    scored_sorted_dec = scores_sorted[::-1]

    # Extract top 'k' indices
    top_k = scored_sorted_dec[0:num_users] 

    return scores[top_k] 

if __name__=='__main__':
    data_file = 'movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user = 'John Carson'
    print "\nUsers similar to " + user + ":\n"
    similar_users = find_similar_users(data, user, 3) 
    print "User\t\t\tSimilarity score\n"
    for item in similar_users:
        print item[0], '\t\t', round(float(item[1]), 2)
