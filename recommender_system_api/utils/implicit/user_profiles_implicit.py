import numpy as np
import pandas as pd


def triple_user_profiles(account_id, gender, model, rating_df, item_id=None):
    if item_id: # Using list of vendor_id from content-based as input neural network
        item_id_country_id = rating_df[rating_df.item_id.isin(item_id)][['actual_item_id', 'item_id', 'item_country_id']]\
            .drop_duplicates().to_numpy()
    else:
        item_id_country_id = rating_df[['actual_item_id', 'item_id', 'item_country_id']].drop_duplicates().to_numpy()

    actual_item_id = item_id_country_id[:, 0]
    item_id = item_id_country_id[:, 1]
    item_country_id = item_id_country_id[:, 2]

    account_id = rating_df[rating_df.actual_account_id == account_id].account_id.iloc[0]
    account_id = np.full_like(item_id, account_id)
    gender = np.full_like(item_id, gender)

    predictions = model.predict([account_id, gender, item_id, item_country_id])
    predictions = np.array([a[0] for a in predictions])

    predicted_rating = pd.DataFrame({'actual_item_id': actual_item_id, 'triplet_rating': predictions})

    return predicted_rating