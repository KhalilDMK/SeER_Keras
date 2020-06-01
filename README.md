# SeER_Keras
Keras implementation of the paper "Sequence-based Explainable Hybrid Song Recommendation".

This is an anonymous copy of the private Github repository for RecSys '20 submission.

## Environment settings
We use Keras 2.2.4.

## Description
This repository includes the code necessary to:
* Train SeER and W-SeER and tune their hyperparameters:
Run "Train.py". The code is set up to train W-SeER with its optimal hyperparameters. You can tune the hyperparameters by updating the file. Also, you can change the "model_type" variable's value to "SeER" to train SeER. The model will train and output the RMSE, MAE, MAP@K and NDCG@K results on the test set for every epoch.
* Generate explained recommendations:
Run "Recommendation_Explanation.py". You can update the code to choose the trained model weights to use. We provide the weights of the best performing SeER and W-SeER models that are reported in the paper. The code outputs "topK" recommendations for the chosen user "user_number". Each recommendation is accompanied with a text explanation that includes the start and end times (in μs) of the most important portion of the recommended song to the user. The code is set up to generate explained recommendations using the model from the W-SeER weight file. You can change the code to use the SeER weight file.

<b>Note: </b>You first need to unzip the following files for the code to run properly:
* Data/midi_array.zip
* Data/Time_array.zip
* Trained models/SeER_GRU_latent_150_seq2600_weights.zip
* Trained models/W-SeER_LSTM_latent_150_seq500_weights.zip

## Dataset
We provide the final preprocessed dataset ready to use as input to the model
* midi_array.txt: Flattened midi array. Shape = (6442 songs, 160,000 = 5,000 normalized time steps * 16 channels * 2 (note, velocity))
* triplets.txt: (user, song, rating) triplets. Includes 941,044 ratings of 32,180 users to 6,442 songs.
* song_information.csv: Includes song metadata (song_id, artist_name, title, release, year, duration)
* song_to_number_matching.csv: Matches the song numbers in our dataset to their corresponding song_ids.
* time_array.txt: Array that includes the actual times in μs of each time step of every song in the dataset.
