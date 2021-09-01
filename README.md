# Music Recommender System

## ISI Delhi ( Remote ) Summer Internship Project
### By Umesh Yadav ( 2018UCS0078 ), CSE Department, IIT Jammu
### Under Supervision of Prof. Samir Kumar Neogy, Prof. & Head, ISI Delhi

A music recommender system (research internship project), developed by applying various recommendation algorithms on Million Song Dataset (MSD).

In this Project, I have used (a Subset) of *Million Song Dataset (MSD)*, Provided by [turi.com](https://static.turi.com/datasets/millionsong/10000.txt) which is taken from a subset of *The Echo Nest Taste Profile Subset* from [here](http://millionsongdataset.com/tasteprofile). It contains triplets of **(user_id, song_id, listen_count)** and has been provided in a text file with triplets being seperated by *'/t' (tab)*.
Song metadata consisting of features related to song wasn't available anywhere except the official [site](http://millionsongdataset.com) of *MSD*. I have downloaded the summary file of the whole 280GB Dataset. It was in .h5 format and I used utility ( edited it alot for my own specific requirements ) from [GitHub](https://github.com/AGeoCoder/Million-Song-Dataset-HDF5-to-CSV) to convert that md5 file into .csv file. It contains metadata for **Unique 10k Songs** contained in triplets file of MSD.

I have used this dataset to train and evaluate the Recommender System using four Algorithms:
1. **Popularity Model:** Sort the Songs according to their Popularity Score.
2. **KNN Model:** Aggregate Similar Songs according to their features or Metadata.
3. **Collaborative Filtering ( Memory Based ) Model:**
    1. **Item-Item CF Model:** Find Similar Songs according to the No. of Users who have rated/ listened them
        1. **Jaccard Based Score:** Sorted Songs according to the Jaccard Index/ Score ( More Info [here](https://en.wikipedia.org/wiki/Jaccard_index) or given in the report of this project )
        2. **Conditional Probability Score:** Sorted according to the Score obtained using Conditional Probability ( More Info [here](http://www.ke.tu-darmstadt.de/events/PL-12/papers/08-aiolli.pdf) or given in the report of this project )
    2. **User-User CF Model:** Find Similar Songs according to the No. of Songs listened by Similar Users
        1. **Jaccard Based Score:** Sorted Songs according to the Jaccard Index/ Score ( More Info [here](https://en.wikipedia.org/wiki/Jaccard_index) or given in the report of this project )
        2. **Conditional Probability Score:** Sorted according to the Score obtained using Conditional Probability ( More Info [here](http://www.ke.tu-darmstadt.de/events/PL-12/papers/08-aiolli.pdf) or given in the report of this project )
4. **Latent Factors ( SVD ) Model:** Find the latent factors of User and Songs by using SVD on User_Song Rating (Implicit Rating) Matrix.

So, **Our Objective using above Models / Algorithms is to return an *Ordered (Ranked) List of Songs* that we call Recommendation List using the Listening History of the User** and Obviously, we must not recommend a user, the songs which he/she has already listened to and hence, our Recommendation List contains only Recommendations for our User with no Songs already listened by them.

# Instructions

———————————————

Code Folder Contains the Code for this Project

The main Code is written inside of Jupyter Notebook .ipynb format file. Running it requires some Pre Computed Data which can be made available by me through Google Drive ( It’s more than 10 GB in Size ).

The Report, The Presentation and the CodeBook ( prepared by converting .ipynb code file into pdf ) are in the root folder of this project folder.

The Root Folder Also contains utility script.py for utility functions, rec.py for recommenders classes, Evaluation.py for evaluation metrics functions ( keep these in the same directory with Jupyter Notebook ).
Also, Some other .py files are there which were used in handling the dataset.

Data Folder contains the dataset files ( mostly in .pkl format ) and Original Dataset files in the original folder. 

List Folder contains the precomputed mAP values and Tuning Results for Graphical Representation.

Graphs Folder contains the Final Results Graphs Stored inside the Folder as .png files.

The Whole Recommender System can be run through Jupyter Notebook / Kaggle Kernel(s) / Google Colab.

——————————————————
 Copyright 2020, Umesh Yadav
——————————————————

### License
This project is distributed under [MIT license](https://opensource.org/licenses/MIT). Any feedback, suggestions are higly appreciated.
