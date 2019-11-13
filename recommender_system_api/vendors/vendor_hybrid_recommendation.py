import os
import pandas as pd
from recommender_system_api.vendors.user_profiles import build_user_profile_cb, build_user_profile_nn
from recommender_system_api.utils.implicit.user_profiles_implicit import triple_user_profiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: If new_user: popularity, elif no_rating: implicit feedback/content_based, else: neural_net+content_based
# TODO: Session-based RecSys


def recommended_for_you(user_id, gender, vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model, triplet_model,
                        implicit_vendor_df):
    """
    Show recommendation list when open app. This not based on specific item.
    :param gender: gender of the user
    :param user_id: user_id
    :param new_user: is new user?
    :param has_reviewed: The user have reviewed?
    :return: Recommendation list
    """

    if user_id not in rating_df.user_id.unique():
        vendor_recommendation = triple_user_profiles(account_id=user_id, gender=gender, model=triplet_model,
                                                     rating_df=implicit_vendor_df)\
            .sort_values(by='triplet_rating', ascending=False)

        vendor_recommendation = vendor_recommendation.actual_item_id.to_numpy()
    else:

        # vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model = processing_ouput()
        vendor_recommendation_cb = build_user_profile_cb(user_id, vendor_id_arr=vendor_id_arr,
                                                         cosine_sim=vendor_cosine_sim, rating_df=rating_df)

        vendor_recommendation_nn = build_user_profile_nn(user_id, gender, model=neural_net_model, rating_df=rating_df)

        vendor_recommendation = pd.concat([vendor_recommendation_nn, vendor_recommendation_cb], join='outer', axis=1).fillna(0)

        vendor_recommendation['avg_rating'] = (vendor_recommendation['content_rating'] +
                                               vendor_recommendation['neural_net_rating'])/2

        vendor_recommendation.sort_values(by='avg_rating', ascending=False, inplace=True)

        vendor_recommendation = vendor_recommendation.index.to_numpy()

    return vendor_recommendation


def specific_recommendation(user_id, gender, vendor_id, vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model,
                            triplet_model, implicit_vendor_df):
    """
    If user not rating:
        Use vendor_id to get list of similar vendors from cosine similarity.
    Else:
        Use vendor_id to get list of similar vendors from cosine similarity, then use that list as input of neural net.

    :param gender: gender of the user
    :param user_id:
    :param vendor_id:
    :return:
    """

    if user_id not in rating_df.user_id.unique():
        vendor_recommendations = build_user_profile_cb(user_id, vendor_id_arr, cosine_sim=vendor_cosine_sim,
                                                      rating_df=rating_df, vendor_id=vendor_id)

        if len(implicit_vendor_df[implicit_vendor_df.item_id.isin(vendor_recommendations)].item_id.unique()) > 10:

            vendor_recommendation = triple_user_profiles(account_id=user_id, gender=gender, model=triplet_model,
                                                         rating_df=implicit_vendor_df, item_id=vendor_recommendations).\
                sort_values(by='triplet_rating', ascending=False)

            vendor_recommendations = vendor_recommendation.actual_item_id.to_numpy()
    else:
        vendor_recommendation_cb = build_user_profile_cb(user_id, vendor_id_arr, cosine_sim=vendor_cosine_sim,
                                                         rating_df=rating_df, vendor_id=vendor_id)

        vendor_recommendation = build_user_profile_nn(user_id, gender, model=neural_net_model, rating_df=rating_df,
                                                      vendor_id=vendor_recommendation_cb).sort_values('neural_net_rating', ascending=False)

        vendor_recommendations = vendor_recommendation.index.to_numpy()

    return vendor_recommendations
