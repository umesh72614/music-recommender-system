'''

Umesh Yadav (2018UCS0078), CSE Department, IIT JMU
contact: 2018ucs0078@iitjammu.ac.in

This code contains a set of classes and methods of various
Recommender Algorithms used in Music Recommender System.

This is part of the Music Recommender System project
(Dataset as Subset of Million Song Dataset) from My ISI Delhi
2020 ( Remote ) Summer Internship.

Copyright 2020, Umesh Yadav

NOTE: Proper credits must be given to the author of this repository
while using this code in any of the concerned research/ project work.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''

# Import Dependencies
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import  NearestNeighbors
from collections import OrderedDict
from collections import Counter
# from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
# from operator import itemgetter

# Class for Popularity based Recommender System model
class popularity_recommender():
    
    def __init__(self):
        self.train_data = None
        self.train_u2s = None
        self.popularity_recommendations = None
    
    # Create the popularity based recommender system model
    def create(self, train_util_dict):
        self.train_data = train_util_dict["dataset"]
        self.train_u2s = train_util_dict["u2s"]

        # Get Total listen count for each song
        train_data_grouped = self.train_data.groupby(['song_id','title']).agg({'listen_count': 'count'}).reset_index()
        
        # Calculate Total listen count
        grouped_sum = train_data_grouped['listen_count'].sum()
        
        # Calculate Score as Percentage of listen count for each song
        train_data_grouped['score']  = train_data_grouped['listen_count'].div(grouped_sum)*100
        
        # Sort the songs based upon recommendation score
        self.popularity_recommendations = train_data_grouped.sort_values(['score', 'song_id'], ascending = [0,1])
    
    # Use the popularity based recommender system model to make recommendations
    def recommend(self, user, tau=500, return_type="df"):
        recommendations = self.popularity_recommendations
        
        # Remove Songs which User has already listened or rated
        user_songs = self.train_u2s[user]
       
        # Below code search for songs from user_recommendations df into user_songs list
        # and return a boolean column or list using .isin() func.
        # https://stackoverflow.com/questions/27965295/dropping-rows-from-dataframe-based-on-a-not-in-condition
       
        recommendations_df = recommendations[ ~recommendations["song_id"].isin(user_songs)]
        
        # Get the top tau recommendations
        recommendations_df = recommendations_df.head(tau)
        
        
        if (return_type == "list"):
            return list(recommendations_df["song_id"])
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
            return recommendations_df
            
        elif (return_type == "ordered_dict"):
            
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            recommendations_ordered_dict = OrderedDict(zip(recommendations_df["song_id"], recommendations_df["score"]))
            return recommendations_ordered_dict
    
    
# Class for KNN Similarity Based Recommender Model
class knn_recommender():
    
    def __init__(self):
        self.train_songs_metadata = None
        self.train_data = None
        self.train_songs_list = None
        self.train_s2u = None
        self.train_u2s = None
        self.train_s2t = None
        self.train_s2i = None
        self.n_neighbors = None
        self.distances = None
        self.indices = None
        
    # Create the KNN based recommender system model
    def create(self, train_songs_metadata, train_util_dict, n_neighbors=10):
        self.train_songs_metadata = train_songs_metadata
        self.train_data = train_util_dict["dataset"]
        self.train_songs_list = train_util_dict["songs"]
        self.train_s2u = train_util_dict["s2u"]
        self.train_u2s = train_util_dict["u2s"]
        self.train_s2t = train_util_dict["s2t"]
        self.train_s2i = train_util_dict["s2i"]
        self.n_neighbors = n_neighbors
        # Apply KNN to get Indices and Distances of n_neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(train_songs_metadata)
        self.distances, self.indices = nbrs.kneighbors(train_songs_metadata)
        
    # Get Similar songs to a given song
    def get_similar_items(self, item, return_type="df"):
    
        # Get song index and then list of indices of similar songs
        item_index = self.train_s2i[item]
        # Drop index of the song itself since it's most similar to itself
        # so not worth recommending / similarity comparison
        similar_songs_indices = self.indices[item_index][1:]
        
        if (return_type == "list"):
        
            # Create Ordered list of Similar Songs
            similar_songs = [self.train_songs_list[i] for i in similar_songs_indices]
            return similar_songs
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create DataFrame of Similar Songs with their title and scores
            similar_songs = [self.train_songs_list[i] for i in similar_songs_indices]
            similar_songs_titles = [self.train_s2t[song] for song in similar_songs]
            score = self.distances[item_index][1:]
            return pd.DataFrame({"song_id":similar_songs, "title":similar_songs_titles, "score":score})
            
        elif (return_type == "ordered_dict"):
        
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            similar_songs = [self.train_songs_list[i] for i in similar_songs_indices]
            score = self.distances[item_index][1:]
            recommendations_ordered_dict = OrderedDict(zip(similar_songs,score))
            return recommendations_ordered_dict
        
    # Use KNN Recommender to recommend Songs to user
    def recommend(self, user, tau=500, return_type="df"):
        
        # Get list of unique songs listened by user
        user_songs = self.train_u2s[user]
        
        recommendations_list = []
        recommend_dict = {}
        
        # Loop through the songs listened by user
        for item in user_songs:
            
            # Get similar songs similar to the listened song
            similar_dict = self.get_similar_items(item, return_type="ordered_dict")
            
            # Assign each similar song a score = -ve of it's distance
            for recommend in similar_dict.keys():
                score = -similar_dict[recommend]
                if recommend in recommend_dict:
                    recommend_dict[recommend] += score
                else:
                    recommend_dict[recommend] = score
        
        # Sort the songs in decreasing order of their score
        recommendations_list = sorted(recommend_dict.keys(), key=lambda s:recommend_dict[s], reverse=True)
        
        # Recommend only top tau songs
        recommendations_list = recommendations_list[:tau]
          
        if (return_type == "list"):
            return recommendations_list
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create a DataFrame of Recommended (Song, Title, Score) Triplets

            recommendations_songs_titles = [self.train_s2t[song] for song in recommendations_list]
            score = [recommend_dict[song] for song in recommendations_list]
            return pd.DataFrame({"song_id":recommendations_list, "title":recommendations_songs_titles, "score":score})
            
        elif (return_type == "ordered_dict"):
        
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            score = [recommend_dict[song] for song in recommendations_list]
            return OrderedDict(zip(recommendations_list,score))


# Class for SVD Latent Factor Based Recommender Model
class svd_recommender():

    def __init__(self):
        self.M = None
        self.train_data = None
        self.train_songs_list = None
        self.train_u2s = None
        self.train_s2t = None
        self.train_s2i = None
        self.train_u2i = None
        self.U = None
        self.Vt = None
        self.R = None
        self.min_R = None
        self.k = None
        self.svd_recommendations = None
     
    # Create the SVD based recommender system model
    def create(self, M, train_util_dict, k=1300):
        self.train_data = train_util_dict["dataset"]
        self.train_songs_list = train_util_dict["songs"]
        self.train_u2s = train_util_dict["u2s"]
        self.train_s2t = train_util_dict["s2t"]
        self.train_s2i = train_util_dict["s2i"]
        self.train_u2i = train_util_dict["u2i"]
        self.k = k
        self.M = M
        self.U, _, self.Vt = svds(self.M, self.k)
        self.R = np.matmul(self.U, self.Vt)
        self.min_R = np.amin(self.R)
        
    # Use SVD Recommender to recommend Songs to user
    def recommend(self, user, tau=500, return_type = "df"):
     
        # Get index of User and Indices of Songs listened by User
        user_index = self.train_u2i[user]
        user_songs_index = list(map(lambda x: self.train_s2i[x], self.train_u2s[user]))
        
        # Set Scores of Songs listened by User = min(R)-1 so that
        # After sorting in reverse order, they get accumulated in the last
        # and since, (tau <<< no. of songs) and hence they don't get recommended
        self.R[user_index,user_songs_index] = self.min_R-1.0
        self.min_R = self.min_R-1.0
        recommend_index_list = (-self.R[user_index,:]).argsort()
        
        # Recommend only top tau songs
        recommend_index_list = recommend_index_list[:tau]
        
        if (return_type == "list"):
            return list(np.array(self.train_songs_list)[recommend_index_list])
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create a DataFrame of Recommended (Song, Title, Score) Triplets
            recommendations_list = list(np.array(self.train_songs_list)[recommend_index_list])
            recommendations_songs_titles = [self.train_s2t[song] for song in recommendations_list]
            score = self.R[user_index, recommend_index_list]
            return pd.DataFrame({"song_id":recommendations_list, "title":recommendations_songs_titles, "score":score})
            
        elif (return_type == "ordered_dict"):
        
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            recommendations_list = [self.train_songs_list[index] for index in recommend_index_list]
            score = self.R[user_index, recommend_index_list]
            recommendations_ordered_dict = OrderedDict(zip(recommendations_list,score))
            return recommendations_ordered_dict

# Class for CF (collaborative filtering) Based Recommender Model
# This is created using Matrix Method for Evaluation Purposes
class cf_recommender():

    def __init__(self, type, method):
        self.train_data = None
        self.train_songs_list = None
        self.train_s2u = None
        self.train_u2s = None
        self.train_s2t = None
        self.train_s2i = None
        self.train_u2i = None
        self.type = type
        self.method = method
        self.M = None
        self.nume = None
        self.U = None
        self.S = None
        self.R = None
        self.Q = None
        self.A = None
    
    # Create the cf recommender system model
    def create(self, M, train_util_dict, nume=None, S_jaccard=None, S_prob=None, U_jaccard=None, U_prob=None, R=None, i=None, j=None, sim=False, A=0, Q=1):
        self.train_data = train_util_dict["dataset"]
        self.train_songs_list = train_util_dict["songs"]
        self.train_users_list = train_util_dict["users"]
        self.train_s2u = train_util_dict["s2u"]
        self.train_u2s = train_util_dict["u2s"]
        self.train_s2t = train_util_dict["s2t"]
        self.train_s2i = train_util_dict["s2i"]
        self.train_u2i = train_util_dict["u2i"]
        self.M = M.toarray()   # We can skip / comment this line if we want to work with matrices not numpy arrays
        self.M = M
        self.R = R
        if self.type == "item_item":
            if self.method == "jaccard":
                self.S = S_jaccard
            elif self.method == "prob":
                self.S = S_prob
        elif self.type == "user_user":
            if self.method == "jaccard":
                self.U = U_jaccard
            elif self.method == "prob":
                self.U = U_prob
        self.Q = Q
        self.A = A
        self.nume = None
        
        # If you want self.S/ self.U to be Computed if not given then sim=True
        
        # if self.R is None or self.S is None or self.U is None:
        if self.R is None or sim:
        
            # i: set of user_indices / user_index to be considered for train/test
            if i is None:
                i = list(range(len(self.train_users_list)))
            
            # j: set of song_indices / song_index to be considered for train/test
            if j is None:
                j = list(range(len(self.train_songs_list)))
            
            if self.type == "item_item":
                
                if self.method == "jaccard":
                    
                    if S_jaccard is None:
                    
                        if nume is None:
                            # Below step requires high RAM usage
                            self.nume = np.dot(self.M.T[j,:],self.M)
                        else:
                            self.nume = nume
                        
                        n = self.M.shape[1]
                        r = np.sum(self.M.T, axis=1)
                        r.shape = (n,1)
                        # the code below is step wise so as to consume less RAM
                        r[j] = r[j]**A
                        self.nume = self.nume/r[j]
                        r = np.sum(self.M.T, axis=1)
                        r = r**(1-A)
                        r.shape = (1,n)
                        self.nume = self.nume/r
                        self.S = self.nume**Q
                        # can also use direct below instead of above multiple
                        # but it will require more/ high RAM usage
                        # self.S = nume / ((r[j]**A)*(r.T**(1-A)))
                    
                    else:
                        self.S = S_jaccard
                    
                elif self.method == "prob":
                
                    if S_prob is None:
                    
                        if nume is None:
                            # Below step requires high RAM usage
                            self.nume = np.dot(self.M.T[j,:],self.M)
                        else:
                            self.nume = nume
                        
                        n = self.M.shape[1]
                        r = np.sum(self.M.T, axis=1)
                        r.shape = (n,1)
                        # Below step requires high RAM usage
                        self.S = self.nume / (r[j] + r.T - self.nume)
                        
                    else:
                        self.S = S_prob
                
                if R is None:
                    # Below step requires high RAM usage
                    self.R = np.matmul(self.M[i,j], self.S)  # np.matmul only works with np arrays not matrices, for them, use np.dot
                
            if self.type == "user_user":
            
                if self.method == "jaccard":
                    
                    if U_jaccard is None:
                    
                        if nume is None:
                            # Below step requires high RAM usage
                            self.nume = np.dot(self.M[i,:],self.M.T)
                        else:
                            self.nume = nume
                        
                        m = self.M.shape[0]
                        r = np.sum(self.M, axis=1)
                        r.shape = (m,1)
                        # the code below is step wise so as to consume less RAM
                        r[i] = r[i]**A
                        self.nume = self.nume/r[i]
                        r = np.sum(self.M, axis=1)
                        r = r**(1-A)
                        r.shape = (1,m)
                        self.nume = self.nume/r
                        self.U = self.nume**Q
                        # can also use direct below instead of above multiple
                        # but it will require more/ high RAM usage
                        # self.U = nume / ((r[j]**A)*(r.T**(1-A)))
                    
                    else:
                        self.U = U_jaccard
                    
                elif self.method == "prob":
                
                    if U_prob is None:
                    
                        if nume is None:
                            # Below step requires high RAM usage
                            self.nume = np.dot(self.M[i,:],self.M.T)
                        else:
                            self.nume = nume
                        
                        m = self.M.shape[0]
                        r = np.sum(self.M, axis=1)
                        r.shape = (m,1)
                        # Below step requires high RAM usage
                        self.U = self.nume / (r[i] + r.T - self.nume)
                    
                    else:
                        self.U = U_prob
                
                if R is None:
                    # Below step requires high RAM usage
                    self.R = np.matmul(self.U,self.M)  # np.matmul only works with np arrays not matrices, for them, use np.dot
    
    # Get items similar to a given item
    def get_similar_items(self, item, item_index=None, return_type="df"):
                
        if (self.type == "item_item"):
            
            # Get index of Song
            if item_index is None:
                item_index = self.train_s2i[item]
                
            # Get Similar Songs by Sorting Songs in reverse order as per score
            similar_items_indices = (-self.S[item_index,:]).argsort()
            
            # Drop index of the song itself since it's most similar to itself
            # so not worth recommending / similarity comparison
            similar_items_indices = similar_items_indices[1:]
            
            # Generate Similar Songs (and their scores) using their indices
            score = self.S[item_index,similar_items_indices]
            similar_items = [self.train_songs_list[index] for index in similar_items_indices]
        
        elif self.type == "user_user":
        
            # Get index of User
            if item_index is None:
                item_index = self.train_u2i[item]
                
            # Get Similar Songs by Sorting Songs in reverse order as per score
            similar_items_indices = (-self.U[item_index,:]).argsort()
            
            # Drop index of the user itself since it's most similar to itself
            # so not worth recommending / similarity comparison
            similar_items_indices = similar_items_indices[1:]
            
            # Generate Similar Songs (and their scores) using their indices
            score = self.U[item_index,similar_items_indices]
            similar_items = [self.train_users_list[index] for index in similar_items_indices]
        
        if (return_type == "list"):
            return similar_items
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create a DataFrame of Recommended (Song, Title, Score) Triplets
            if (self.type == "item_item"):
                similar_songs_titles = [self.train_s2t[song] for song in similar_items]
                return pd.DataFrame({'song_id':similar_items, 'title':similar_songs_titles, 'score':score})
            elif self.type == "user_user":
                return pd.DataFrame({'user_id':similar_items, 'score':score})
                
        elif (return_type == "ordered_dict"):
            
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            recommendations_ordered_dict = OrderedDict(zip(similar_items, score))
            return recommendations_ordered_dict
                
    # Use CF Recommender to recommend Songs to user
    def recommend(self, user, user_index=None, tau=500, return_type="df"):
        
        # Get index of User
        if user_index is None:
            user_index = self.train_u2i[user]
        
        # Get Indices of Songs listened by User
        user_songs_index = list(map(lambda x: self.train_s2i[x], self.train_u2s[user]))
        recommend_index_list = []
        score = []

        # Set Scores of Songs listened by User = min(R)-1 so that
        # After sorting in reverse order, they get accumulated in the last
        # and since, (tau <<< no. of songs) and hence they don't get recommended
        self.R[user_index,user_songs_index] = 0
        recommend_index_list = (-self.R[user_index,:]).argsort()
        score = self.R[user_index, recommend_index_list]
        
        # Recommend only top tau songs
        score = score[:tau]
        recommend_index_list = recommend_index_list[:tau]
        recommendations = [self.train_songs_list[i] for i in recommend_index_list]

        if (return_type == "list"):
            return recommendations
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create a DataFrame of Recommended (Song, Title, Score) Triplets
            titles = [self.train_s2t[song] for song in recommendations ]
            return pd.DataFrame({'song_id':recommendations, 'title':titles, 'score':score})
            
        elif (return_type == "ordered_dict"):
            
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            recommendations_ordered_dict = OrderedDict(zip(recommendations,score))
            return recommendations_ordered_dict
            

# Class for CF (collaborative filtering) Based Recommender Model
# This method is very slow
# This is created using loop Method for Testing Purposes
class cf_sim_recommender():

    def __init__(self, type, method):
        self.train_data = None
        self.train_songs_list = None
        self.train_s2u = None
        self.train_u2s = None
        self.train_s2t = None
        self.type = type
        self.method = method
        self.Q = None
        self.A = None

    # Create the cf recommender system model
    def create(self, train_util_dict, A=0, Q=1):
        self.train_data = train_util_dict["dataset"]
        self.train_songs_list = train_util_dict["songs"]
        self.train_s2u = train_util_dict["s2u"]
        self.train_s2t = train_util_dict["s2t"]
        self.train_u2s = train_util_dict["u2s"]
        self.Q = Q
        self.A = A

    # Utility function to compute the intersection
    def intersect(self,l, m):
        c = Counter(l + m)
        return sum(v == 2 for v in c.values())

    # Utility function to compute the similarity score for two given songs
    def match(self, song, user_song):
        l1=len(self.train_s2u[song])
        l2=len(self.train_s2u[user_song])
        intersection = self.intersect(self.train_s2u[song], self.train_s2u[user_song])
        up = float(intersection)
        if up>0:
            if self.method == "prob":
                dn = math.pow(l1,self.A)*math.pow(l2,(1.0-self.A))   #self.A = 0.15
            elif self.method == "jaccard":
                dn = l1 + l2 - up
            return up/dn
        return 0.0

    # Utility function to create a dictionary of similar songs and their scores
    def score(self, user_songs, train_songs_list, item=None): #calculate Wuv/Wij
        song_scores= {}
        if self.type == "item_item":
            for song in train_songs_list:
                song_scores[song] = 0.0
                if song not in self.train_s2u:
                    continue
                for user_song in user_songs:
                    if user_song not in self.train_s2u:
                        continue
                    if user_song == song:
                        continue
                    song_match = self.match(song, user_song)
                    song_scores[song] += math.pow(song_match, self.Q)        #self.Q = 3
            return song_scores, None
        elif self.type == "user_user":
            user_scores = {}
            for user in self.train_u2s:
                if user == item:
                    continue
                intersection = self.intersect(user_songs, self.train_u2s[user])
                w = float(intersection)
                if w > 0:
                    l1 = len(user_songs)
                    l2 = len(self.train_u2s[user])
                    if self.method == "prob":
                        w /= (math.pow(l1,self.A)*(math.pow(l2,(1.0-self.A))))
                    elif self.method == "jaccard":
                        w /= (l1 + l2 - w)
                    w = math.pow(w,self.Q)
                    user_scores[user] = w
                for song in self.train_u2s[user]:
                    if song in user_songs:
                        continue
                    if song in song_scores:
                        song_scores[song]+=w
                    else:
                        song_scores[song]=w
            return song_scores, user_scores

    # Get similar items to a given item
    def get_similar_items(self, item, return_type="df"):

        if (self.type == "item_item"):

            scores, _ = self.score([item], self.train_songs_list)
            similar_items = sorted(scores.keys(),key=lambda s:scores[s],reverse=True)

        elif self.type == "user_user":
            _, scores = self.score(self.train_u2s[item], self.train_songs_list, item=item)
            similar_items = sorted(scores.keys(),key=lambda s:scores[s],reverse=True)

        if (return_type == "list"):
            return similar_items

        # By Default Return DataFrame

        elif (return_type == "df"):

            # Create a DataFrame of Recommended (Song, Title, Score) Triplets
            score = [scores[item] for item in similar_items]
            if (self.type == "item_item"):
                similar_songs_titles = [self.train_s2t[song] for song in similar_items]
                return pd.DataFrame({'song_id':similar_items, 'title':similar_songs_titles, 'score':score})
            else:
                return pd.DataFrame({'user_id':similar_items, 'score':score})

        elif (return_type == "ordered_dict"):

            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            score = [scores[item] for item in similar_items]
            recommendations_ordered_dict = OrderedDict(zip(similar_items, score))
            return recommendations_ordered_dict

    # Use CF Recommender to recommend Songs to user
    def recommend(self, user, tau=500, return_type="df"):

        scores = {}
        ssongs=[]
        if user in self.train_u2s:
            if self.type == "item_item":
                scores, _ = self.score(self.train_u2s[user], self.train_songs_list)
            elif self.type == "user_user":
                scores, _ = self.score(self.train_u2s[user], self.train_songs_list,item=user)
            ssongs = sorted(scores.keys(),key=lambda s:scores[s],reverse=True)
        else:
            ssongs = list(self.train_songs_list)

        cleaned_songs = ssongs[:tau]

        if (return_type == "list"):
            return cleaned_songs

        # By Default Return DataFrame

        elif (return_type == "df"):

            # Create a DataFrame of Recommended (Song, Title, Score) Triplets
            score = [scores[song] for song in cleaned_songs]
            cleaned_songs_titles = [self.train_s2t[song] for song in cleaned_songs]
            return pd.DataFrame({'song_id':cleaned_songs, 'title':cleaned_songs_titles, 'score':score})

        elif (return_type == "ordered_dict"):

            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            score = [scores[song] for song in cleaned_songs]
            recommendations_ordered_dict = OrderedDict(zip(cleaned_songs,score))
            return recommendations_ordered_dict
