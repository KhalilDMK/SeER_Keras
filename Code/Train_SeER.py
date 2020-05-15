import numpy as np
from time import time
from Code.Data_train import Dataset_train
from Code.SeER import get_model
from Code.Evaluate import prepare_data_for_evaluation, root_mean_squared_error_evaluation, mean_absolute_error_evaluation, map_at_k_evaluation
from keras.optimizers import Adam, sgd

# Define hyperparameters

triplets_filename = '../Data/triplets'
midi_array_filename = '../Data/midi_array'
num_latent_features = 150
learning_rate = 0.001
optimizer = 'Adam'
batch_size = 500
num_epochs = 20
sequence_length = 2600
num_channels = 16 #Choose between 1 and 16
topK = 10
interaction_threshold = 3 #Choose between 1 and 5

# Main

if __name__ == '__main__':

    # Read data
    dataset = Dataset_train(triplets_filename, midi_array_filename, sequence_length)
    train_set, test_set, midi_array, num_users, num_songs, test_true_labels = dataset.read_files

    # Reduce number of channels to num_channels
    midi_array = dataset.reduce_num_channels(midi_array, num_channels)

    # Define model
    model = get_model(num_users, num_songs, num_latent_features, midi_array, num_channels, sequence_length)

    #Compile model
    model.compile(optimizer=eval(optimizer)(lr=learning_rate), loss='mean_squared_error', metrics=['mse', 'mae'])

    #Train model
    for epoch in range(num_epochs):
        print('Epoch ' + str(epoch + 1))

        # Training
        train_users, train_songs, train_labels = dataset.generate_train_instances(train_set)
        hist = model.fit([np.array(train_users), np.array(train_songs)], train_labels, batch_size=batch_size, epochs=1,
                         verbose=1, shuffle=True)

        # Evaluation
        t1 = time()
        pred_df, predictions = prepare_data_for_evaluation(test_set, test_true_labels, model, interaction_threshold, batch_size)
        RMSE = root_mean_squared_error_evaluation(predictions, test_true_labels)
        MAE = mean_absolute_error_evaluation(predictions, test_true_labels)
        MAP_at_K = map_at_k_evaluation(pred_df, topK)
        t2 = time()
        print('RMSE = ' + str(RMSE) + ' MAE = ' + str(MAE) + ' MAP@' + str(topK) + ': ' + str(MAP_at_K) + '\nEvaluation time: ' + str(t2 - t1) + ' s')