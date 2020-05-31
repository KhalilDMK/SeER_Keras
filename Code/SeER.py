from keras.layers import Embedding, Input, Flatten, LSTM, Reshape, dot, Dropout, GRU, SimpleRNN
from keras.models import Model

#Method to create the model

def get_model(num_users, num_songs, num_latent_features, midi_array, num_channels, sequence_length):
    user_input = Input(shape=(1,), dtype='float32', name = 'user_input')
    embedding_User = Embedding(input_dim=num_users, output_dim=num_latent_features, name='embedding_user', embeddings_initializer='normal', input_length=1)
    user_latent = Flatten()(embedding_User(user_input))
    song_input = Input(shape=(1,), dtype='float32', name = 'song_input')
    flat_song_feature_input = Embedding(input_dim=num_songs, output_dim=sequence_length*num_channels*2, weights=[midi_array], name='flat_song_feature_input', input_length=1, trainable=False)(song_input)
    song_feature_input = Reshape((sequence_length, num_channels*2), name='song_feature_input')(flat_song_feature_input)
    LSTM_1 = GRU(num_latent_features, name='LSTM_1', dropout=0.2, recurrent_dropout=0.2)(song_feature_input)
    #dropout_1 = Dropout(0.2)(LSTM_1)
    prediction = dot([LSTM_1, user_latent], axes=-1, name='prediction')
    model = Model(inputs=[user_input, song_input], outputs=prediction)
    return model

#method to create the explainability model

def get_explainability_model(num_users, num_latent_features, num_channels, model_name):
    user_input = Input(shape=(1,), dtype='float32', name = 'user_input')
    embedding_User = Embedding(input_dim=num_users, output_dim=num_latent_features, name='embedding_user', embeddings_initializer='normal', input_length=1)
    user_latent = Flatten()(embedding_User(user_input))
    song_array_input = Input(shape = (None, num_channels*2), name='song_array_input')
    LSTM_1 = GRU(num_latent_features, name='LSTM_1', dropout=0.2, recurrent_dropout=0.2)(song_array_input)
    #dropout_1 = Dropout(0.2)(LSTM_1)
    prediction = dot([LSTM_1, user_latent], axes=-1, name='prediction')
    explainability_model = Model(inputs=[user_input, song_array_input], outputs=prediction)
    explainability_model.load_weights(model_name + '_weights.h5', by_name=True)
    return explainability_model