import numpy as np
import pandas as pd

#Method to generate recommendations with explanations

def generate_recommendations_with_explanations_for_user(len_sections, user_number, midi_array, num_channels, model, explainability_model, song_information, time_array, num_songs, topK, batch_size, song_to_number_matching):
    test_user, test_song = create_input_for_user(user_number, num_songs)
    recommendations = generate_recommendations_for_user(test_user, test_song, model, topK, batch_size, song_information, song_to_number_matching)
    recommendations = explainability(recommendations, len_sections, user_number, midi_array, num_channels, explainability_model, song_information, time_array)
    return recommendations

#Method to create input with user and all songs

def create_input_for_user(user_number, num_songs):
    print('Creating test input for user...')
    test_user = np.array([user_number] * num_songs)
    test_song = np.array(range(num_songs))
    return test_user, test_song

#Method to generate recommendations for user

def generate_recommendations_for_user(test_user, test_song, model, topK, batch_size, song_information, song_to_number_matching):
    print('Generating recommendations for user...')
    predictions = model.predict([test_user, test_song], batch_size=batch_size, verbose=0)
    predictions = np.array([x[0] for x in predictions])
    recommendations = pd.DataFrame({'song': test_song, 'rating': list(predictions)})
    recommendations.sort_values('rating', axis=0, inplace=True, ascending=False)
    recommendations = recommendations.reset_index(drop=True)
    recommendations = recommendations[:topK]
    recommendations['song_id'] = [match_song_number_to_song_id(x, song_to_number_matching) for x in recommendations['song']]
    recommendations = pd.merge(recommendations, song_information, on=['song_id'])
    return recommendations

#Method to match song numbers to song ids

def match_song_number_to_song_id(song_number, song_to_number_matching):
    song_id = song_to_number_matching['song_id'][song_number]
    return song_id

#Method to generate explainability for user

def explainability(recommendations, len_sections, user_number, midi_array, num_channels, explainability_model, song_information, time_array):
    print('Generating explanations for top recommended songs...')
    recommendations['Explanation'] = [''] * len(recommendations)
    for i in range(len(recommendations)):
        song = recommendations['song'][i]
        song_id = recommendations['song_id'][i]
        most_relevant_start, most_relevant_end = most_relevant_portion_of_song(len_sections, song_id, song, user_number, midi_array, num_channels, explainability_model, song_information, time_array)
        recommendations['Explanation'][i] = (most_relevant_start, most_relevant_end)
    return recommendations

#Method to determine most relevant portion of song

def most_relevant_portion_of_song(len_sections, song_id, song, user_number, midi_array, num_channels, explainability_model, song_information, time_array):
    explainability_user_input, explainability_song_input, start_indices, end_indices = create_explainability_input_for_user_and_song(len_sections, song_id, song, user_number, midi_array, num_channels, song_information, time_array)
    predictions = explainability_model.predict([explainability_user_input, explainability_song_input], batch_size=1, verbose=1)
    predictions = np.array([float(predictions[i]) for i in range(len(predictions))])
    most_relevant_portion = np.argmax(predictions)
    most_relevant_start = time_array[song, start_indices[most_relevant_portion]]
    most_relevant_end = time_array[song, end_indices[most_relevant_portion]]
    return most_relevant_start, most_relevant_end


# Method to create explainability input for user and song

def create_explainability_input_for_user_and_song(len_sections, song_id, song, user_number, midi_array, num_channels, song_information, time_array):
    # Sliding window sampling
    start_indices, end_indices = generate_sections_with_sliding_window(len_sections, song_id, song, song_information, time_array)
    explainability_user_input = [user_number] * len(start_indices)
    max_length = max([end_indices[i] - start_indices[i] + 1 for i in range(len(start_indices))])
    explainability_song_input = np.array([np.reshape(
        np.pad(midi_array[song][start * num_channels * 2 + 1:end * num_channels * 2],
               (0, max_length * 32 - len(midi_array[song][start * num_channels * 2 + 1:end * num_channels * 2])),
               'constant'), (-1, num_channels * 2)) for (start, end) in zip(start_indices, end_indices)])
    return explainability_user_input, explainability_song_input, start_indices, end_indices

#Method to generate sections with sliding window

def generate_sections_with_sliding_window(len_sections, song_id, song, song_information, time_array):
    sample_start_times = list(range(1, int(song_information[song_information['song_id'] == song_id]['duration'].values[0] - len_sections)))
    start_indices = [[i for i in range(len(time_array[song])) if time_array[song][i] / 1000000 >= j][0] if [i for i in range(len(time_array[song])) if time_array[song][i] / 1000000 >= j] != [] else np.nan for j in sample_start_times]
    end_indices = [[i for i in range(len(time_array[song])) if time_array[song][i] / 1000000 <= j + len_sections and time_array[song][i] != 0][-1] if [i for i in range(len(time_array[song])) if time_array[song][i] / 1000000 <= j + len_sections and time_array[song][i] != 0] != [] else np.nan for j in sample_start_times]
    nan_indices = [x[0] for x in list(np.argwhere(np.isnan(np.array(start_indices)))) + list(np.argwhere(np.isnan(np.array(end_indices))))]
    start_indices = [start_indices[i] for i in list(range(len(start_indices))) if i not in nan_indices]
    end_indices = [end_indices[i] for i in list(range(len(end_indices))) if i not in nan_indices]
    return start_indices, end_indices

