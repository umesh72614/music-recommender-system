''' 

Umesh Yadav (2018UCS0078), CSE Department, IIT JMU
contact: 2018ucs0078@iitjammu.ac.in

This code contains a set of functions used for evaluating the 
Music Recommender System with the help of mean average precision at tau(=500).

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

import numpy as np
import random

# Find AP_at_tau for given list of predictions
def aptau(actual, predicted, tau=500):
    if len(predicted)>tau:            # AP_at_tau
        predicted = predicted[:tau]

    score = 0.0
    num_hits = 0.0
    
    if not actual:
        return 1.0

    for i,p in enumerate(predicted):
        # Assuming Predicted not contains repeated recommendations
        # otherwise uncomment the below commented segment of the code
        if p in actual: #and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)     # give more weight to top recommendataions

    return score / min(len(actual), tau)

# Randomly select given percentage of elements
# from a given list
def create_test_sample(list_a, percentage):
    print("Length of user_test_and_training:%d" % len(list_a))
    k = int(len(list_a) * percentage)
    random.seed(0)
    indicies = random.sample(range(len(list_a)), k)
    new_list = [list_a[i] for i in indicies]
    print("Length of user sample:%d" % len(new_list))
    return new_list

# Create mAP_at_tau vs sampling rate list
def sample_list_map(model, test_users, test_util_dict, tau=500):
    sample_list = [0.25, 0.5, 0.75, 1]
    list_samples = []
    for perc in sample_list:
        list_sample = create_test_sample(test_users, perc)
        list_samples.append(list_sample)
    map_sample_list = []
    for user_list in list_samples:
        AP_at_tau = 0
        for i,user in enumerate(user_list):
            predicted = model.recommend(user, return_type="list")
            actual = test_util_dict["u2s"][user]
            AP_at_tau += aptau(actual, predicted, tau)
            if i%10000 == 0:
                print(i+1, AP_at_tau/(i+1)*100, "%")
        mAP_at_tau = AP_at_tau/(i+1)*100
        print(i+1, mAP_at_tau, "%")
        map_sample_list.append(mAP_at_tau)
        
    return sample_list, map_sample_list
    
# Find mAP_at_tau for given test_list and Sampling Rate
def maptau(model, test_users, test_util_dict, sampling_rate=0.3, tau=500):
    # Prepare the test Sample with given Sampling Rate
    test_sample = create_test_sample(test_users, sampling_rate)

    # Calcuate mAP_at_tau for test Sample
    AP_at_tau = 0
    for i,user in enumerate(test_sample):
        predicted = model.recommend(user, return_type="list")
        actual = test_util_dict["u2s"][user]
        AP_at_tau += aptau(actual, predicted, tau)
        if i%10000 == 0:
            print(i+1, AP_at_tau/(i+1)*100, "%")
    mAP_at_tau = AP_at_tau/(i+1)*100
    print(i+1, mAP_at_tau, "%")
    
    return mAP_at_tau

# Utility function to generate test sample for User_User_CF Models
def gen_newtest(newtrain_data, newtrain_util_dict, common_test_sample, test_util_dict):
    newtest_sample = list(set(common_test_sample).intersection(newtrain_data["user_id"].unique()))
    ## since we are using set in above, so order might get changed
    newtest_u2s = {}
    for user in newtest_sample:
        songs = test_util_dict["u2s"][user]
        for song in songs:
            if song in newtrain_util_dict["newtrain_data"]["songs"]:
                if user in newtest_u2s:
                    newtest_u2s[user] += [song]
                else:
                    newtest_u2s[user] = [song]
