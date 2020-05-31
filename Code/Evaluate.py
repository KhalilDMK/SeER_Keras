import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import math

# Method to prepare the data for evaluation

def prepare_data_for_evaluation(test_set, test_true_labels, model, interaction_threshold, batch_size):
    predictions = model.predict([[x[0] for x in test_set], [x[1] for x in test_set]], batch_size=batch_size, verbose=1)
    predictions = np.array([x[0] for x in predictions])
    pred_df = pd.DataFrame(
        {'user': [x[0] for x in test_set], 'song': [x[1] for x in test_set], 'rating': list(predictions),
         'true': test_true_labels})
    pred_df['relevant'] = np.where(pred_df['true'] >= interaction_threshold, 1, 0)
    pred_df['rank'] = pred_df.groupby('user')['rating'].rank(method='first', ascending=False)
    pred_df['rank_true'] = pred_df.groupby('user')['true'].rank(method='first', ascending=False)
    pred_df.sort_values(['user', 'rank'], inplace=True)
    return pred_df, predictions

# Method to calculate RMSE

def root_mean_squared_error_evaluation(predictions, test_true_labels):
    return np.sqrt(np.mean(np.square(predictions - np.array(test_true_labels))))

# Method to calculate MAE

def mean_absolute_error_evaluation(predictions, test_true_labels):
    return mean_absolute_error(np.array(test_true_labels), predictions)

# Method to calculate MAP@K

def map_at_k_evaluation(pred_df, topk):
    AP = 0.0
    counter = 0
    for i in pred_df['user'].unique():
        counter += 1
        user_df = pred_df[pred_df['user'] == i]
        user_df.sort_values('rating', axis=0, inplace=True, ascending=False)
        top_N_items = user_df['relevant'].values[:topk]
        p_list = []
        for j in range(1, len(top_N_items) + 1):
            l = user_df['relevant'].values[:j]
            val = np.sum(l) / len(l)
            p_list.append(val)
        sum_val = sum(p_list * top_N_items)
        if (sum(user_df['relevant'] > 0)):
            AP = AP + sum_val / sum(user_df['relevant'])
    MAP = AP / pred_df['user'].nunique()
    return MAP

#Method to calculate NDCG@k

def ndcg_at_k_evaluation(pred_df, top_k):
    topp_k = pred_df[pred_df['rank_true'] <= top_k].copy()
    topp_k['idcg_unit'] = topp_k['rank_true'].apply(
        lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
    topp_k['idcg'] = topp_k.groupby(['user'])['idcg_unit'].transform('sum')

    test_in_top_k = topp_k[topp_k['rank'] <= top_k].copy()
    test_in_top_k['dcg_unit'] = test_in_top_k['rank'].apply(
        lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
    test_in_top_k['dcg'] = test_in_top_k.groupby(['user'])['dcg_unit'].transform('sum')
    test_in_top_k['ndcg'] = test_in_top_k['dcg'] / topp_k['idcg']
    ndcg_sum = test_in_top_k.groupby(['user']).apply
    ndcg = np.sum(test_in_top_k.groupby(['user'])['ndcg'].max()) / len(pred_df['user'].unique())
    del (topp_k, test_in_top_k)
    return ndcg