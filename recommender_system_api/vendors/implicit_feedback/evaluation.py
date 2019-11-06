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


def average_roc_auc(match_model, train_data, test_data):
    """
    Compute the ROC AUC for each user and average over users
    :param match_model:
    :param train_data:
    :param test_data:
    :return:
    """
    max_user_id = max(train_data['account_id'].max(), test_data['account_id'].max())
    max_vendor_id = max(train_data['vendor_id'].max(), test_data['vendor_id']. max())
    user_auc_scores = []
    for account_id in range(1, max_user_id + 1):
        positive_vendor_train = train_data[train_data['account_id'] == account_id]
        positive_vendor_test = test_data[test_data['account_id'] == account_id]

        # Consider all the items already seen in the training set
        all_vendor_ids = np.arange(1, max_vendor_id + 1)
        # Return the items that are not in the training set
        vendors_to_rank = np.setdiff1d(all_vendor_ids, positive_vendor_train['vendor_id'].to_numpy())

        # Ground truth: return 1 for each item positively present in the test set and 0 otherwise
        expected = np.isin(vendors_to_rank, positive_vendor_test['vendor_id'].to_numpy())

        if np.sum(expected) >= 1:
            # At least on positive test value to rank
            repeated_user_id = np.empty_like(vendors_to_rank)
            repeated_user_id.fill(account_id)

            predicted = match_model.predict([repeated_user_id, vendors_to_rank], batch_size=64)

            user_auc_scores.append(roc_auc_score(expected, predicted))

    return sum(user_auc_scores) / len(user_auc_scores)


def sample_triplets(train_df, n_vendors, random_seed=0):
    """
    Sample negatives at random
    :param train_df:
    :param n_vendors:
    :param random_seed:
    :return:
    """
    neg_random_state = np.random.RandomState(random_seed)
    user_ids = train_df['account_id'].values
    positive_vendor_ids = train_df['vendor_id'].values
    negative_vendor_ids = neg_random_state.choice(np.setdiff1d(np.arange(1, n_vendors), positive_vendor_ids),
                                                  size=len(user_ids))

    return [user_ids, positive_vendor_ids, negative_vendor_ids]
