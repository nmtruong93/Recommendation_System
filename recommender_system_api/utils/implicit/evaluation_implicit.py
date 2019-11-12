import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np


def identity_loss(y_true, y_pred):
    """
    Ignore y_true and return mean of y_predict
    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.reduce_mean(y_pred + 0 * y_true)


def margin_comparator_loss(inputs, margin=1.0):
    """
    Comparator loss for a pair of precomputed similarities

    If the inputs are cosine similarities, they each have range in (-1, 1), therefore their difference have range in
    (-2, 2). Using a margin of 1. can therefore make sense.

    If the input similarities are not normalized, it can be beneficial to use larger values for the margin of
    comparator loss
    :param inputs:
    :param margin:
    :return:
    """
    positive_pair_sim, negative_pair_sim = inputs
    return tf.maximum(negative_pair_sim - positive_pair_sim + margin, 0)


def average_roc_auc(match_model, train_data, test_data, dataset_df):
    """
    Compute the ROC AUC for each user and average over users
    :param match_model:
    :param train_data:
    :param test_data:
    :return:
    """
    max_user_id = max(train_data['account_id'].max(), test_data['account_id'].max())
    max_item_id = max(train_data['item_id'].max(), test_data['item_id']. max())
    user_auc_scores = []
    for account_id in range(1, max_user_id + 1):
        # Gender corresponding account_id
        gender = dataset_df.loc[dataset_df.account_id == account_id].gender.values[0]

        positive_item_train = train_data[train_data['account_id'] == account_id]
        positive_item_test = test_data[test_data['account_id'] == account_id]

        # Consider all the items already seen in the training set
        all_item_ids = np.arange(1, max_item_id + 1)
        # Return the items that are not in the training set
        items_to_rank = np.setdiff1d(all_item_ids, positive_item_train['item_id'].to_numpy())
        item_country_ids = []
        for i in items_to_rank:
            item_country_id = dataset_df.loc[dataset_df.item_id == i].item_country_id.values[0]
            item_country_ids.append(item_country_id)
        item_country_ids = np.array(item_country_ids)

        # Ground truth: return 1 for each item positively present in the test set and 0 otherwise
        expected = np.isin(items_to_rank, positive_item_test['item_id'].to_numpy())

        if np.sum(expected) >= 1:
            # At least on positive test value to rank
            repeated_user_id = np.full_like(items_to_rank, account_id)
            repeated_gender = np.full_like(items_to_rank, gender)

            predicted = match_model.predict([repeated_user_id, repeated_gender, items_to_rank, item_country_ids],
                                            batch_size=64)

            user_auc_scores.append(roc_auc_score(expected, predicted))

    return sum(user_auc_scores) / len(user_auc_scores)


def sample_triplets(train_df, n_items, dataset_df, random_seed=0):
    """
    Sample negatives at random
    :param train_df:
    :param n_items:
    :param random_seed:
    :return:
    """
    neg_random_state = np.random.RandomState(random_seed)
    user_ids = train_df['account_id'].to_numpy()
    gender = train_df['gender'].to_numpy()
    positive_item_ids = train_df['item_id'].to_numpy()
    positive_item_country_id = train_df['item_country_id'].to_numpy()

    negative_item_ids = neg_random_state.choice(np.setdiff1d(np.arange(1, n_items), positive_item_ids),
                                                  size=len(user_ids))
    negative_item_country_ids = []
    for i in negative_item_ids:
        item_country_id = dataset_df.loc[dataset_df.item_id == i].item_country_id.values[0]
        negative_item_country_ids.append(item_country_id)

    negative_item_country_ids = np.array(negative_item_country_ids)
    return [user_ids, gender, positive_item_ids, negative_item_ids, positive_item_country_id, negative_item_country_ids]
