"""

Credit:
    Alexis Greenstreet (October 4, 2015) University of Wisconsin-Madison
    https://github.com/AGeoCoder/Million-Song-Dataset-HDF5-to-CSV

I, Umesh Yadav (2018UCS0078), IIT JMU, have edited the code from above link ( a utility
function for extracting specific features from .h5 file format ) according to my
own specific requirements and needs for my project. The original code belongs to the
owner and credits/ link has been provided above for the same.

This code is designed to convert the HDF5 files of the Million Song Dataset
to a CSV by extracting various song properties.

The script writes to a "SongCSV.csv" in the directory containing this script.

Please note that in the current form, this code only extracts the following
information/ features from the HDF5 files:
song_id, release, release_7digitalid, artist_id, artist_name, artist_hotttnesss,
duration, key_confidence, loudness, mode_confidence, song_hotttnesss, tempo,
time_signature_confidence, title, track_id,year

While Extracting the Features I found that:-
Following Features aren't avilable to extract:
    bars_confidence
    beats_confidence
    sections_confidence
    segments_confidence
    sections_confidence
    tatums_confidence
Following Features are always 0 so not useful
    danceability
    energy

This file also requires the use of "hdf5_getters.py", written by
Thierry Bertin-Mahieux (2010) at Columbia University
( can be found on Github although I have also made it available in Project folder )


"""

import hdf5_getters
import pickle

class Song:
   def __init__(self, songID):
       self.id = songID
       self.albumName = None
       self.albumID = None
       self.artistID = None
       self.artistName = None
       self.artistHotttnesss = None
       self.duration = None
       self.keyConfidence = None
       self.loudness = None
       self.modeConfidence = None
       self.songHotttnesss = None
       self.tempo = None
       self.timeSignatureConfidence = None
       self.title = None
       self.trackID = None
       self.year = None

# Utility for Searching Features using Binary Search
def binary_search(a, v):
    l = 0
    h = len(a) - 1
    m = 0
    while l <= h:
        m = (h + l) // 2
        # Check if v is present at m
        if a[m] < v:
            l = m + 1
        # If v is greater, move to right
        elif a[m] > v:
            h = m - 1
        # If v is smaller, move to left
        else:
            return m
    # If v is not present
    return -1

def main():
    
    # Path to output file
    o = r'./data/SongCSV.csv'
    outputFile1 = open(o, 'w')
    csvRowString = ""

    # Path to utility dict to get list of song ids
    # NOTE: since, no. of songs are same in data, train and test data;
    # so, utility dict of train (train_util_dict.pkl)/ test (test_util_dict.pkl) data can also be used
    # if utility script is not in data folder, it can be generated using utility_script.py
    data_path = r'./data/data_util_dict.pkl'
    dataset = pickle.load(open(data_path, "rb"))
    # store song ids in sorted form to use binary search
    ID = sorted(list(dataset["song_id"].unique()))

    #################################################
    #change the order of the csv file here
    #Default is to list all available attributes (in alphabetical order)
    
    csvRowString = ("song_id,release,release_7digitalid,artist_id,artist_name,artist_hotttnesss,duration,key_confidence,loudness,mode_confidence,song_hotttnesss,tempo,time_signature_confidence,title,track_id,year")
    
    features = csvRowString.split(",")
    print("Features to be Extracted are:",len(features))
    for i in range(len(features)):
        print(i+1,features[i])

    #################################################

    outputFile1.write(csvRowString + "\n")
    csvRowString = ""

    ##############################################
    
    # Path to input .h5 file
    # Please Note that .h5 and .H5 are different (case sensitive here)
    # Currently i have deleted this file to save space
    # You can download "msd_summary_file.h5" from MSD Website using this link:
    # http://millionsongdataset.com/pages/getting-dataset/
    
    f = r'./data/msd_summary_file.h5'
    counter = 0
    songH5File = hdf5_getters.open_h5_file_read(f)
    n = hdf5_getters.get_num_songs(songH5File)
    print("No. of Songs in .h5 file:",n)
    unique_song_ids = dict()

    for i in range(n):
        if (counter == 10000):
            break
        songID = str(hdf5_getters.get_song_id(songH5File,i)).strip("b'")
        if (unique_song_ids.setdefault(songID,False)):
            continue
        unique_song_ids[songID] = True
        result = binary_search(ID,songID)
        if (result == -1 ):
            continue
        counter += 1
        print(counter,songID,"has been found.")
        song = Song(songID)
        song.artistID = str(hdf5_getters.get_artist_id(songH5File,i)).strip("b'""")
        song.albumID = str(hdf5_getters.get_release_7digitalid(songH5File,i))
        song.albumName = str(hdf5_getters.get_release(songH5File,i)).strip("b'""")
        song.artistName = str(hdf5_getters.get_artist_name(songH5File,i)).strip("b'""")
        song.artistHotttnesss = str(hdf5_getters.get_artist_hotttnesss(songH5File,i))
        # song.danceability = str(hdf5_getters.get_danceability(songH5File,i))
        song.duration = str(hdf5_getters.get_duration(songH5File,i))
        # song.energy = str(hdf5_getters.get_energy(songH5File,i))
        song.keyConfidence = str(hdf5_getters.get_key_confidence(songH5File,i))
        song.loudness = str(hdf5_getters.get_loudness(songH5File,i))
        song.modeConfidence = str(hdf5_getters.get_mode_confidence(songH5File,i))
        song.songHotttnesss = str(hdf5_getters.get_song_hotttnesss(songH5File,i))
        song.tempo = str(hdf5_getters.get_tempo(songH5File,i))
        song.timeSignatureConfidence = str(hdf5_getters.get_time_signature_confidence(songH5File,i))
        song.title = str(hdf5_getters.get_title(songH5File,i)).strip("b'""")
        song.trackID = str(hdf5_getters.get_track_id(songH5File,i)).strip("b'""")
        song.year = str(hdf5_getters.get_year(songH5File,i))

        for attribute in features:

            if attribute == 'release_7digitalid':
                csvRowString += song.albumID
            elif attribute == 'release':
                albumName = song.albumName
                albumName = albumName.replace(',',"")
                csvRowString += "\"" + albumName + "\""
            elif attribute == 'artist_id':
                csvRowString += "\"" + song.artistID + "\""
            elif attribute == 'artist_name':
                csvRowString += "\"" + song.artistName + "\""
            elif attribute == 'artist_hotttnesss':
                csvRowString += "\"" + song.artistHotttnesss + "\""
            elif attribute == 'duration':
                csvRowString += song.duration
            elif attribute == 'key_confidence':
                csvRowString += song.keyConfidence
            elif attribute == 'loudness':
                csvRowString += song.loudness
            elif attribute == 'mode_confidence':
                csvRowString += song.modeConfidence
            elif attribute == 'song_id':
                csvRowString += "\"" + song.id + "\""
            elif attribute == 'song_hotttnesss':
                csvRowString += "\"" + song.songHotttnesss + "\""
            elif attribute == 'mode_confidence':
                csvRowString += "\"" + song.modeConfidence + "\""
            elif attribute == 'tempo':
                csvRowString += song.tempo
            elif attribute == 'time_signature_confidence':
                csvRowString += song.timeSignatureConfidence
            elif attribute == 'title':
                csvRowString += "\"" + song.title + "\""
            elif attribute == 'track_id':
                csvRowString += "\"" + song.trackID + "\""
            elif attribute == 'year':
                csvRowString += song.year
            else:
                csvRowString += "Erm. This didn't work. Error. :( :(\n"
            # Don't add "," at the end of each Row in the csv
            if (attribute != features[-1]):
                csvRowString += ","
        csvRowString += "\n"

    outputFile1.write(csvRowString)
    csvRowString = ""
    print("Process Successfully Completed !")
    print(counter,"out of 10k songs features have been saved to ",o)
    print("Out of",i+1,"songs from Million Songs")

    songH5File.close()
    outputFile1.close()


if __name__ == '__main__':
    main()
