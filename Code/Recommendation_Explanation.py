from Code.Data_test import Dataset_test
from Code.Recommendation_utils import generate_recommendations_with_explanations_for_user

# Define hyperparameters
model_name = '../Trained models/SeER_LSTM_latent_150_seq500'
midi_array_filename = '../Data/midi_array'
song_to_number_matching_filename = '../Data/song_to_number_matching'
song_information_filename = '../Data/song_information'
time_array_filename = '../Data/time_array'
triplets_filename = '../Data/triplets'
sequence_length = 500
batch_size = 500
topK = 5  # Number of recommendations
len_sections = 10  # in seconds
num_channels = 16
num_latent_features = 150

model_type = 'SeER'
global get_explainability_model, get_model
exec('from Code.' + model_type + ' import get_explainability_model, get_model')

# Define test user
user_number = 4

# Main

if __name__ == '__main__':

    # Read files and create models
    dataset = Dataset_test(midi_array_filename, time_array_filename, song_to_number_matching_filename, song_information_filename, triplets_filename, model_name, num_latent_features, num_channels, sequence_length, get_explainability_model, get_model)
    midi_array, time_array, song_to_number_matching, num_songs, song_information, num_users, num_songs, model, explainability_model = dataset.read_files_and_create_models

    #Generate recommendations with explanations for user_number
    recommendations = generate_recommendations_with_explanations_for_user(len_sections, user_number, midi_array, num_channels, model, explainability_model, song_information, time_array, num_songs, topK, batch_size, song_to_number_matching)

    #Show topK recommendations
    print(recommendations.to_string())