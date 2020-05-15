# SeER_Keras
Keras implementation of the paper "SeER: An Explainable Deep Learning MIDI-based Hybrid Song Recommender System"

Authors: Khalil Damak and Olfa Nasraoui

Link to paper: https://arxiv.org/abs/1907.01640

\bold{Please cite the paper if you use our code. Thanks!}

## Environment settings
We use Keras 2.2.4.

## Description
This repository includes the code necessary to:
* Train SeER and tune its hyperparameters:
Run "Train_SeER.py". You can tune the hyperparameters by updating the file. The model will train and output the RMSE, MAE and MAP@K results on the test set for every epoch.
* Generate explained recommendations:
Run "Recommendation_Explanation_SeER.py". You can update the code to choose the trained model weights to use. We provide the weights of the best performing model with GRU and 150 latent factors. The code outputs "topK" recommendations for the chosen user "user_number". Each recommendation is accompanied with a text explanation that includes the start and end times (in μs) of the most important portion of the recommended song to the user.

\bold{Note: }You first need to unzip the following files for the code to run properly:
* Data/midi_array.zip
* Data/Time_array.zip
* Trained models/model_GRU_latent_150_weights.zip

## Dataset
We provide the final preprocessed dataset ready to use as input to the model
* midi_array.txt: Flattened midi array. Shape = (6442 songs, 160000 = 5,000 normalized time steps * 16 channels * 2 (note, velocity))
* triplets.txt: (user, song, rating) triplets. Includes 941,044 ratings of 32,180 users to 6,442 songs.
* song_information.csv: Includes song metadata (song_id, artist_name, title, release, year, duration)
* song_to_number_matching.csv: Matches the song numbers in our datasets to their song_ids.
* time_array.txt: Array that includes the actual times in μs of each time step of every song in the dataset.
