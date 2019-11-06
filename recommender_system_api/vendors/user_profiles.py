import numpy as np
import pandas as pd


def build_user_profile_nn(user_id, gender, model, rating_df, vendor_id=None):
    if vendor_id: # Using list of vendor_id from content-based as input neural network
        vendor_id_country_id = rating_df[rating_df.vendor_id.isin(vendor_id)][['vendor_id', 'vd_country_id']]\
            .drop_duplicates().to_numpy()
        vendor_id = vendor_id_country_id[:, 0]
        vd_country_id = vendor_id_country_id[:, 1]
    else:
        vendor_id_country_id = rating_df[['vendor_id', 'vd_country_id']].drop_duplicates().to_numpy()
        vendor_id = vendor_id_country_id[:, 0] # Recommend on the home page
        vd_country_id = vendor_id_country_id[:, 1]

    user_id = rating_df[rating_df.user_id == user_id].user_id.iloc[0]
    user_id = np.array([user_id for i in range(len(vendor_id))])
    gender = np.array([gender for i in range(len(vendor_id))])
    predictions = model.predict([user_id, gender, vendor_id, vd_country_id])
    predictions = np.array([a[0] for a in predictions])
    predicted_rating = pd.DataFrame({'vendor_id': vendor_id, 'neural_net_rating': predictions})

    return predicted_rating.set_index('vendor_id')


# TODO: Improve performance of this function, consider remove vendor_df
def build_user_profile_cb(user_id, vendor_id_arr, cosine_sim, rating_df, vendor_id=None):
    """
    Build user profile with rating prediction
    :param vendor_id: vendor id
    :param user_id: integer - user_id that we need to build profile
    :param vendor_df: DataFrame - output from vendor_coupon_processing function
    :param cosine_sim: cosine similarity matrix
    :return: rating corresponding to index of cosine similarity or vendor_df
    """
    vendor_df = pd.DataFrame(vendor_id_arr, columns=['vendor_id'])
    if vendor_id:  # User has not rating, recommend based on content only.
        indices = pd.Series(vendor_df.vendor_id)
        idx = indices[indices == vendor_id].index[0]
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False).index
        vendors_recommendation = [indices[indices.index == i].iloc[0] for i in score_series]
    else:   # User has rating, recommend based on user profile
        vendors_rated_by_user = rating_df[rating_df.user_id == user_id][['vendor_id', 'rating']]
        id_of_rated_vendors = vendors_rated_by_user.vendor_id.unique()

        # DataFrame with 3 columns ['index of count_vector', 'vendor_id', 'rating']
        rated_vendors = vendor_df[vendor_df.vendor_id.isin(id_of_rated_vendors)].reset_index() \
            .merge(vendors_rated_by_user, on='vendor_id', how='inner') \
            .rename(columns={'index': 'cosine_sim_index'})[['cosine_sim_index', 'vendor_id', 'rating']]

        rated_cosine_indices = rated_vendors.cosine_sim_index.to_numpy()
        # Cosine similarity between rated vendors versus all vendors
        rated_cosine_similarity = pd.DataFrame(cosine_sim[rated_cosine_indices], index=rated_cosine_indices)
        # Rating of rated vendors * cosine similarity between rated vendors versus all vendors
        similar_vendors_rating = pd.DataFrame(rated_vendors[['rating']].to_numpy() * rated_cosine_similarity.to_numpy(),
                                              index=rated_cosine_similarity.index, columns=rated_cosine_similarity.columns)

        user_profile = (similar_vendors_rating.sum(axis=0)/rated_cosine_similarity.sum(axis=0)).fillna(0)
        # Remove vendors already rated.
        user_profile = user_profile[~user_profile.index.isin(rated_cosine_indices)].sort_values(ascending=False)

        vendors_recommendation = pd.Series(user_profile.to_numpy(), index=vendor_df.iloc[user_profile.index].vendor_id,
                                           name='content_rating')

    return vendors_recommendation
