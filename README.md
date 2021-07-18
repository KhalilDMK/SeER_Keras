# SeER_Keras
Keras implementation of the paper "Sequence-based Explainable Hybrid Song Recommendation".
Accepted at Frontiers in Big Data, Data Mining and Management.
https://www.frontiersin.org/articles/10.3389/fdata.2021.693494/abstract

## Authors
Khalil Damak, University of Louisville.<br>
Olfa Nasraoui, University of Louisville.<br>
W. Scott Sanders, University of Louisville.

## Abstract
Despite advances in deep learning methods for song recommendation, most existing methods do not take advantage of the sequential nature of song content.
In addition, there is a lack of methods that can explain their predictions using the content of recommended songs and only a few approaches can handle the item cold start problem.
In this work, we propose a hybrid deep learning model that uses collaborative filtering (CF) and deep learning sequence models on the Musical Instrument Digital Interface (MIDI) content of songs to provide accurate recommendations, while also being able to generate a relevant, personalized explanation for each recommended song.
Compared to state-of-the-art methods, our validation experiments showed that in addition to generating explainable recommendations, our model stood out among the top performers in terms of recommendation accuracy and the ability to handle the item cold start problem.
Moreover, validation shows that our personalized explanations capture properties that are in accordance with the user's preferences.

## Environment settings
We use Keras 2.2.4.

## Description
This repository includes the code necessary to:
* <b>Train SeER:</b>
Run "Train.py". The code is set up to train SeER with its optimal hyperparameters. You can tune the hyperparameters by updating the file. The model will train and output the RMSE, MAE, MAP@K and NDCG@K results on the test set for every epoch.
* <b>Generate explained recommendations:</b>
Run "Recommendation_Explanation.py". You can update the code to choose the trained model weights to use. We provide the weights of the best performing SeER model that are reported in the paper. The code outputs "topK" recommendations for the chosen user "user_number". Each recommendation is accompanied with a text explanation that includes the start and end times (in μs) of the most important portion of the recommended song to the user. The code is set up to generate explained recommendations using the model from the SeER weight file.

<b>Note: </b>You first need to unzip the following files for the code to run properly:
* Data/midi_array.zip
* Data/Time_array.zip
* Trained models/SeER_LSTM_latent_150_seq500_weights.zip

## Dataset
We provide the final preprocessed dataset ready to use as input to the model
* midi_array.txt: Flattened midi array. Shape = (6442 songs, 160,000 = 5,000 normalized time steps * 16 channels * 2 (note, velocity))
* triplets.txt: (user, song, rating) triplets. Includes 941,044 ratings of 32,180 users to 6,442 songs.
* song_information.csv: Includes song metadata (song_id, artist_name, title, release, year, duration)
* song_to_number_matching.csv: Matches the song numbers in our dataset to their corresponding song_ids.
* time_array.txt: Array that includes the actual times in μs of each time step of every song in the dataset.
