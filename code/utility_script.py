'''

Umesh Yadav (2018UCS0078), CSE Department, IIT JMU
contact: 2018ucs0078@iitjammu.ac.in

This code contains a set of functions to generate utilities for 
Music Recommender System.

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

# Below is a utility script to generate helpful Functions for the
# Music Recommendation System that I (Umesh Yadav) have Created as a part of my
# ISI Delhi (Remote) Summer Internship 2020


# Import Dependencies
import os.path
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csc_matrix
import time
from datetime import timedelta

# Utility Function to Generate:
# u2s Mapping/ dictionary in given dataset
# s2u Mapping/ dictionary in given dataset
# s2t Mapping/ dictionary in given dataset
# s2i Mapping/ dictionary in given dataset
# u2i Mapping/ dictionary in given dataset
# songs list from given dataset
# users list from given dataset

def gen_dict(type="u2s", dataset=None, dataset_path=None, dataset_name="data", file_path=None, save=True, save_path=None):
    
    if file_path is None:
        file_path = "./data/"+dataset_name.split("_")[0]+"_"+type+".pkl"
        
    if save_path is None:
        save_path = "./data/"+dataset_name.split("_")[0]+"_"+type+".pkl"
    
    if (os.path.isfile(file_path)):
        dicti = pickle.load(open(file_path, 'rb'))
        return dicti
        
    if dataset is None:
        if dataset_path is None:
            dataset_path = "./data/"+dataset_name+".pkl"
        dataset = pd.read_pickle(dataset_path)
 
    d = {}
    
    if type in ["u2s","s2u"]:
        lines = dataset[["user_id","song_id"]].values
        for line in lines:
            user, song = line
            if type == "u2s":
                if user in d:
                    d[user].add(song)
                else:
                    d[user] = set([song])
            elif type == "s2u":
                if song in d:
                    d[song].add(user)
                else:
                    d[song]=set([user])
        for key in d:
            d[key] = list(d[key])
        print("Completed", type, "for", dataset_name)

    elif type == "s2t":
        d = dict(dataset[["song_id","title"]].values)
        print("Completed", type, "for", dataset_name)
        
    elif type == "s2i":
        songs = list(dataset["song_id"].unique())
        d = dict(list(map(lambda x: (x[1],x[0]),enumerate(songs))))
        print("Completed", type, "for", dataset_name)
        
    elif type == "u2i":
        users = list(dataset["user_id"].unique())
        d = dict(list(map(lambda x: (x[1],x[0]),enumerate(users))))
        print("Completed", type, "for", dataset_name)
    
    if save:
        pickle.dump(d, open(save_path, "wb"))
        print("Saved", type, "in:", save_path)
        
    return d
    
# Utility Funciton to Generate Rating Matrix ( Implicit Rating )
# from given Dataset. It return a csc_matrix
    
def gen_M(dataset=None, dataset_path=None, dataset_name="train_data", file_path=None,save=True, save_path=None):
    
    if file_path is None:
        file_path = "./data/"+dataset_name.split("_")[0]+"_M.pkl"
        
    if save_path is None:
        save_path = "./data/"+dataset_name.split("_")[0]+"_M.pkl"
    
    if (os.path.isfile(file_path)):
        M = pickle.load(open(file_path, 'rb'))
        return M
    
    if dataset is None:
        if dataset_path is None:
            dataset_path = "./data/"+dataset_name+".pkl"
        dataset = pd.read_pickle(dataset_path)
        
    users = list(dataset["user_id"].unique())
    songs = list(dataset["song_id"].unique())
    u2s = gen_dict("u2s", dataset, dataset_name=dataset_name, save=False)
    s2i = gen_dict("s2i", dataset, dataset_name=dataset_name, save=False)
    
    M = csc_matrix(np.zeros(shape=(len(users),len(songs)))).tolil()
    for i,user in enumerate(users):
        user_songs_indices = [s2i[song] for song in u2s[user]]
        M[i,user_songs_indices] = 1
        print("Finished", i+1, "user out of", len(users), "users")
    M = csc_matrix(M, dtype=np.float32)
        
    if save:
        pickle.dump(M, open(save_path, "wb"))
        print("Saved M in:", save_path)

    return M
    
# Utility Function to Generate a Utility:
# u2s Mapping/ dictionary in all given datasets
# s2u Mapping/ dictionary in all given datasets
# s2t Mapping/ dictionary in all given datasets
# s2i Mapping/ dictionary in all given datasets
# u2i Mapping/ dictionary in all given datasets
# songs list from given datasets
# users list from given datasets
# all of above in one utility dictionary with key(s) as
# the given name(s) of the datasets
    
def gen_util_dict(dataset_list=None, dataset_path_list=None, dataset_names_list=None, file_path=None, save=True, save_path=None):

    if file_path is None:
        if dataset_names_list and ( len(dataset_names_list) == 1):
            file_path = "./data/"+dataset_names_list[0]+"_util_dict.pkl"
        else:
            file_path = r"./data/util_dict.pkl"
        
    if save_path is None:
        if dataset_names_list and ( len(dataset_names_list) == 1):
            save_path = "./data/"+dataset_names_list[0]+"_util_dict.pkl"
        else:
            save_path = "./data/util_dict.pkl"
        
    if (os.path.isfile(file_path)):
        util_dict = pickle.load(open(file_path, 'rb'))
        return util_dict
    
    if dataset_list is None:
        if dataset_path_list is None:
            data_path = r'./data/data.pkl'
            train_data_path = r'./data/train_data.pkl'
            test_data_path = r'./data/test_data.pkl'
            dataset_path_list = [data_path, train_data_path, test_data_path]
        dataset_list = [pd.read_pickle(path) for path in dataset_path_list]
    
    if dataset_names_list is None:
        dataset_names_list = ["data","train_data","test_data"]
    
    util_dict = {}
    for i, dataset in enumerate(dataset_list):
        dictionary = {}
        for type in ["s2u", "u2s", "s2t", "s2i", "u2i"]:
            dictionary[type] = gen_dict(type, dataset, dataset_name=dataset_names_list[i], save=False)
        dictionary["dataset"] = dataset
        dictionary["songs"] = list(dataset["song_id"].unique())
        dictionary["users"] = list(dataset["user_id"].unique())
        util_dict[dataset_names_list[i]] = dictionary

    if save:
        pickle.dump(util_dict, open(save_path, "wb"))
        print("Saved util_dict in:", save_path)
    
    return util_dict
    

def main():

    start_time = time.time()
    
    print("Script to Generate utility_dict and M")

    # Generate Utility Dictionary
    gen_util_dict()

    # Generate Rating Matrix M
    gen_M()

    elapsed_time_secs = time.time() - start_time

    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

    print(msg)

if __name__ == "__main__":
    main()
