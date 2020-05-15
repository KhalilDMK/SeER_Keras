import numpy as np
import pandas as pd
import json
from keras.models import load_model
from Code.SeER import get_explainability_model, get_model

class Dataset_test(object):

    def __init__(self, midi_array_filename, time_array_filename, song_to_number_matching_filename, song_information_filename, triplets_filename, model_name, num_latent_features, num_channels, sequence_length):
        self.midi_array_filename = midi_array_filename
        self.time_array_filename = time_array_filename
        self.song_to_number_matching_filename = song_to_number_matching_filename
        self.song_information_filename = song_information_filename
        self.triplets_filename = triplets_filename
        self.model_name = model_name
        self.num_latent_features = num_latent_features
        self.num_channels = num_channels
        self.sequence_length = sequence_length

    # Method to read the files and create the models

    @property
    def read_files_and_create_models(self):
        midi_array = read_midi_array(self.midi_array_filename)
        time_array = read_time_array(self.time_array_filename)
        song_to_number_matching, num_songs = read_list_of_songs(self.song_to_number_matching_filename)
        song_information = read_song_information(self.song_information_filename)
        num_users, num_songs = get_number_of_users_and_songs(self.triplets_filename)
        model = load_trained_model(self.model_name, num_users, num_songs, self.num_latent_features, midi_array, self.num_channels, self.sequence_length)
        explainability_model = get_explainability_model(num_users, self.num_latent_features, self.num_channels, self.model_name)
        print('Done.')
        return midi_array, time_array, song_to_number_matching, num_songs, song_information, num_users, num_songs, model, explainability_model

# Method to read the MIDI array

def read_midi_array(midi_array_filename):
    print('Reading MIDI array...')
    with open(midi_array_filename + '.txt') as f:
        midi_array = json.load(f)
    midi_array = np.array(midi_array)
    return midi_array

#Method to read the time array

def read_time_array(time_array_filename):
    print('Reading time array...')
    with open(time_array_filename + '.txt') as f:
        time_array = json.load(f)
    time_array = np.array(time_array)
    return time_array

#Method to read the list of songs

def read_list_of_songs(song_to_number_matching_filename):
    print('Reading list of songs...')
    song_to_number_matching = pd.read_csv(song_to_number_matching_filename + '.csv')[['song_id', 'number']]
    num_songs = len(song_to_number_matching)
    return song_to_number_matching, num_songs

#Method to read the song information

def read_song_information(song_information_filename):
    print('Reading song information...')
    song_information = pd.read_csv(song_information_filename + '.csv')[['song_id', 'artist_name', 'title', 'release', 'year', 'duration']]
    return song_information

def get_number_of_users_and_songs(triplets_filename):
    print('Reading triplets...')
    triplets = pd.read_csv(triplets_filename + '.txt', sep=" ", names=['user', 'song', 'play count'])
    num_users = len(set(triplets['user']))
    num_songs = len(set(triplets['song']))
    return num_users, num_songs

#Method to load the trained model

def load_trained_model(model_name, num_users, num_songs, num_latent_features, midi_array, num_channels, sequence_length):
    print('Reading trained model...')
    model = get_model(num_users, num_songs, num_latent_features, midi_array[:, list(range(sequence_length * 32))], num_channels, sequence_length)
    model.load_weights(model_name + '_weights.h5')
    #model = load_model(model_name + '.h5')
    return model

