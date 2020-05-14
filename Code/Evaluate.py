import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import progressbar
from ml_metrics import mapk

# Method to prepare the data for evaluation

#def prepare_data_for_evaluation(test_set, test_true_labels, model, interaction_threshold, batch_size):
#    predictions = model.predict([[x[0] for x in test_set], [x[1] for x in test_set]], batch_size=batch_size, verbose=1)
#    predictions = np.array([x[0] for x in predictions])
#    pred_df = pd.DataFrame({'user': [x[0] for x in test_set], 'song': [x[1] for x in test_set], 'rating': list(predictions), 'true': test_true_labels})
#    pred_df['relevant'] = np.where(pred_df['true'] >= interaction_threshold, 1, 0)
#    pred_df['rank'] = pred_df.groupby('user')['rating'].rank(method='first', ascending=False)
#    pred_df['rank_true'] = pred_df.groupby('user')['true'].rank(method='first', ascending=False)
#    pred_df.sort_values(['user', 'rank'], inplace=True)
#    return pred_df, predictions

def prepare_data_for_evaluation(test_set, test_true_labels, model, interaction_threshold, batch_size):
    predictions = model.predict([[x[0] for x in test_set], [x[1] for x in test_set]], batch_size=batch_size, verbose=1)
    predictions = np.array([x[0] for x in predictions])
    pred_df = pd.DataFrame({'user': [x[0] for x in test_set], 'song': [x[1] for x in test_set], 'rating': list(predictions), 'true': test_true_labels})
    pred_df['relevant'] = np.where(pred_df['true'] >= interaction_threshold, 1, 0)
    pred_df['recommended'] = np.where(pred_df['rating'] >= interaction_threshold, 1, 0)
    pred_df['relevant_recommendations'] = pred_df['relevant'] * pred_df['recommended']
    return pred_df, predictions

# Method to calculate RMSE

def root_mean_squared_error_evaluation(predictions, test_true_labels):
    return np.sqrt(np.mean(np.square(predictions - np.array(test_true_labels))))

# Method to calculate MAE

def mean_absolute_error_evaluation(predictions, test_true_labels):
    return mean_absolute_error(np.array(test_true_labels), predictions)

# Method to calculate MAP@K

#def map_at_k_evaluation(pred_df, topK):
#    users = list(dict.fromkeys(list(pred_df['user'])))
#    #actual = [list(pred_df[(pred_df['user'] == user) & (pred_df['relevant'] == 1)]['song']) for user in users]
#    actual = [list(pred_df[(pred_df['user'] == user) & (pred_df['rank_true'] <= topK)]['song']) for user in users]
#    predicted = [list(pred_df[(pred_df['user'] == user) & (pred_df['rank'] <= topK)]['song']) for user in users]
#    return mapk(actual, predicted, k=topK)

def map_at_k_evaluation(pred_df, topk):
    AP = 0.0
    counter = 0
    print('Evaluation - MAP: ')
    bar = progressbar.ProgressBar(maxval=len(set(pred_df['user'])), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in pred_df['user'].unique():
        counter += 1
        bar.update(counter)
        user_df = pred_df[pred_df['user'] == i]
        user_df.sort_values('rating', axis=0, inplace=True, ascending=False)
        top_N_items = user_df['relevant_recommendations'].values[:topk]
        p_list = []
        for j in range(1, len(top_N_items) + 1):
            l = user_df['relevant_recommendations'].values[:j]
            val = np.sum(l) / len(l)
            p_list.append(val)
        sum_val = sum(p_list * top_N_items)
        if(sum(user_df['relevant'] > 0)):
            AP = AP + sum_val / sum(user_df['relevant'])
    bar.finish()
    MAP = AP / pred_df['user'].nunique()
    return MAP