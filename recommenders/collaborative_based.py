"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors

# Importing data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
#model=pickle.load(open('SVD.pkl', 'rb'))
with open('resources/models/nlp_model.pkl','rb') as model_file:
    model = pickle.load(model_file)

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    user_ids = pred_movies(movie_list)

    df_init_users = ratings_df[ratings_df['userId'].isin(user_ids)]
    df_init_users =df_init_users.drop_duplicates().reset_index(drop= True)
    movie_ids =[]
    for movie in movie_list:
        movie_ids.append(int(movies_df['movieId'][movies_df['title']==movie]))

    new_row1 = {'userId':1234567, 'movieId':movie_ids[0], 'rating':5}
    new_row2 = {'userId':1234567, 'movieId':movie_ids[1], 'rating':5}
    new_row3 = {'userId':1234567, 'movieId':movie_ids[2], 'rating': 5}
    df_init_users = df_init_users.append([new_row1,new_row2,new_row3],ignore_index=True)

    pivot_user = pd.pivot_table(df_init_users,values='rating',columns='userId',index='movieId')
    pivot_user.fillna(0, inplace=True)
    
    pivot_user = pivot_user.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)), axis=1)

    pivot_user_arr =  np.array(pivot_user)

    cosine_sim=cosine_similarity(pivot_user_arr,pivot_user_arr)
    
    #mat
    m_index_list = list(pivot_user.index)
    df = movies_df[movies_df['movieId'].isin(m_index_list)].reset_index(drop=True)

    #Obtaining indices
    indices = pd.Series(df['title'])
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]

    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)

    # Appending the names of movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    recommended_movies = []
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:10]:
        recommended_movies.append(list(movies_df['title'])[i])
        

    return recommended_movies
