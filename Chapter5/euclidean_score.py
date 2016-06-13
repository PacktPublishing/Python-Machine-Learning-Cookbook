import json
import numpy as np
 
# Returns the Euclidean distance score between user1 and user2 
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # Movies rated by both user1 and user2
    rated_by_both = {} 

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    # If there are no common movies, the score is 0 
    if len(rated_by_both) == 0:
        return 0

    squared_differences = [] 

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))
        
    return 1 / (1 + np.sqrt(np.sum(squared_differences))) 

if __name__=='__main__':
    data_file = 'movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user1 = 'John Carson'
    user2 = 'Michelle Peterson'

    print "\nEuclidean score:"
    print euclidean_score(data, user1, user2) 
