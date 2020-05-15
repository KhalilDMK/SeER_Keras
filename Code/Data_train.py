import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
from sklearn.model_selection import train_test_split

class Dataset_train(object):

    def __init__(self, triplets_filename, midi_array_filename, sequence_length):
        self.triplets_filename = triplets_filename
        self.midi_array_filename = midi_array_filename
        self.sequence_length = sequence_length

    # Method to read the files

    @property
    def read_files(self):

        # Read the files
        print('Reading triplets...')
        triplets = pd.read_csv(self.triplets_filename + '.txt', sep=" ", names=['user', 'song', 'play count'])

        print('Reading MIDI array...')
        with open(self.midi_array_filename + '.txt') as f:
            midi_array = json.load(f)
        midi_array = np.array(midi_array)

        print('Truncating MIDI array to sequence_length...')
        midi_array = midi_array[:, list(range(self.sequence_length * 32))]

        # Calculating the numbers of users and songs
        num_users = len(set(triplets['user']))
        num_songs = len(set(triplets['song']))

        # Splitting triplets into train and test
        print('Splitting triplets into train and test sets...')
        train_set_frame, test_set = train_test_split(triplets, test_size=0.2, random_state=1)
        train_set_frame = train_set_frame.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        test_true_labels = list(test_set['play count'])
        del (test_set['play count'])
        test_set = test_set.values.tolist()
        train_set = sp.dok_matrix((len(set(triplets['user'])) + 1, len(set(triplets['song'])) + 1), dtype=np.float32)
        for row in train_set_frame.values:
            train_set[row[0], row[1]] = row[2]

        print('Done.')

        return train_set, test_set, midi_array, num_users, num_songs, test_true_labels

    # Method to read the files without train/test splitting

    @property
    def read_files_without_splitting(self):

        # Read the files
        print('Reading triplets...')
        triplets = pd.read_csv(self.triplets_filename + '.txt', sep=" ", names=['user', 'song', 'play count'])

        print('Reading MIDI array...')
        with open(self.midi_array_filename + '.txt') as f:
            midi_array = json.load(f)
        midi_array = np.array(midi_array)

        print('Truncating MIDI array to sequence_length...')
        midi_array = midi_array[:, list(range(self.sequence_length * 32))]

        # Calculating the numbers of users and songs
        num_users = len(set(triplets['user']))
        num_songs = len(set(triplets['song']))

        # Splitting triplets into train and test
        print('Creating training set with the whole data...')
        train_set = sp.dok_matrix((len(set(triplets['user'])) + 1, len(set(triplets['song'])) + 1), dtype=np.float32)
        for row in triplets.values:
            train_set[row[0], row[1]] = row[2]

        print('Done.')

        return train_set, midi_array, num_users, num_songs

    # Method to reduce the number of channels

    def reduce_num_channels(self, midi_array, num_channels):
        if num_channels != 16:
            channel_indices = []
            for i in range(2 * num_channels):
                channel_indices += list(range(i, len(midi_array[0]), 32))
            channel_indices = sorted(channel_indices)
            midi_array = midi_array[:, channel_indices]
        return midi_array

    # Method to generate train sequences with negative samples

    def generate_train_instances(self, train_set):
        print('Creating training instances...')
        train_users = train_set.nonzero()[0].tolist()
        train_songs = train_set.nonzero()[1].tolist()
        train_labels = [int(train_set[x, y]) for (x, y) in zip(train_users, train_songs)]
        return train_users, train_songs, train_labels