from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from recommender_system_api.utils.explicit.connections import get_data_by_pandas
from recommender_system_api.utils.explicit import queries
from recommender_system_api.utils.explicit.words_processing import tfidf
from recommender_system_api.vendors.neural_network import create_model, train_model
from recommender_system_api.vendors.vendor_content_based_filtering import vendor_processing, coupon_processing, vendor_coupon_processing
import pickle
from tensorflow.keras.models import load_model
import os
import pandas as pd
import numpy as np
from config.settings import base
from recommender_system_api.vendors.user_profiles import build_user_profile_cb, build_user_profile_nn
from django.core.cache.backends.base import DEFAULT_TIMEOUT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: How to know the new user? New user --> Popularity(balance, distance, rating) <-- New vendors/coupons
# TODO: If new_user: popularity, elif no_rating: implicit feedback/content_based, else: neural_net+content_based
# TODO: Session-based RecSys
# TODO: Hybrid recommendation system using deep learning metadata (user_properties, vendor_properties)

CACHE_TTL = getattr(base, 'CACHE_TTL', DEFAULT_TIMEOUT)


def recommended_for_you(user_id, gender, vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model, triplet_model,
                        implicit_vendor_df, new_user=False):
    """
    Show recommendation list when open app. This not based on specific item.
    :param gender: gender of the user
    :param user_id: user_id
    :param new_user: is new user?
    :param has_reviewed: The user have reviewed?
    :return: Recommendation list
    """

    if user_id in implicit_vendor_df.actual_account_id.unique():
        vendor_recommendation = []
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


def specific_recommendation(user_id, gender, vendor_id, vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model):
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
        vendor_recommendation = build_user_profile_cb(user_id, vendor_id_arr, cosine_sim=vendor_cosine_sim,
                                                      rating_df=rating_df, vendor_id=vendor_id)
        #TODO: Should combine with implicit data
    else:
        vendor_recommendation_cb = build_user_profile_cb(user_id, vendor_id_arr, cosine_sim=vendor_cosine_sim,
                                                         rating_df=rating_df, vendor_id=vendor_id)

        vendor_recommendation = build_user_profile_nn(user_id, gender, model=neural_net_model, rating_df=rating_df,
                                                      vendor_id=vendor_recommendation_cb).sort_values('neural_net_rating', ascending=False)

        vendor_recommendation = vendor_recommendation.index.to_numpy()

    return vendor_recommendation


def load_vendor_models():
    """
    Get the results from all previous processing
    :return:
    """
    model_path = os.path.join(base.BASE_DIR, 'recommender_system_api/models/')
    vendor_cosine_sim = pickle.load(open(os.path.join(model_path, 'vendor_cosine_sim.pickle'), 'rb'))
    vendor_id_arr = pickle.load(open(os.path.join(model_path, 'vendor_id.pickle'), 'rb'))
    neural_net_model = load_model(os.path.join(model_path, 'vendor_neural_net.h5'))
    rating_df = pd.read_feather(os.path.join(model_path, 'rating_df.feather'))
    return vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model


def retrain_vendor_models():

    stopwords_path = os.path.join(base.BASE_DIR, 'recommender_system_api/utils/explicit/')
    model_path = os.path.join(base.BASE_DIR, 'recommender_system_api/models/')

    vendor_df = get_data_by_pandas(query=queries.GET_VENDOR_CONTENT)
    coupon_df = get_data_by_pandas(query=queries.GET_COUPON_CONTENT)
    rating_df = get_data_by_pandas(query=queries.GET_VENDOR_RATING)
    favorite_df = get_data_by_pandas(query=queries.GET_VENDOR_FAVORITE)
    rating_df = pd.concat([rating_df, favorite_df])

    rating_df['gender'] = rating_df.gender.astype('int64')
    rating_df.dropna(inplace=True)
    rating_df['vd_country_id'] = rating_df.vd_country_id.astype('int64')
    rating_df.reset_index(drop=True, inplace=True)

    train_df, test_df = train_test_split(rating_df, test_size=0.2, shuffle=True)
    params = {'n_latent_factors': 10, 'n_users': rating_df.user_id.max(), 'n_vendors': rating_df.vendor_id.max(),
              'n_genders': rating_df.gender.max(), 'n_vendor_countries': rating_df.vd_country_id.max()}
    x_train, y_train = [train_df.user_id, train_df.gender, train_df.vendor_id, train_df.vd_country_id], train_df.rating

    model = create_model(params=params)
    neural_net_model, _ = train_model(x_train=x_train, y_train=y_train, model_path=model_path, model=model)
    neural_net_model.save(os.path.join(model_path, 'vendor_neural_net.h5'))

    score_mean, score_absolute = neural_net_evaluation(neural_net_model, test_df)
    print("=" * 100)
    print("Score mean", score_mean, "Score absolute", score_absolute)

    processed_vendor = vendor_processing(vendor_df)
    processed_coupon = coupon_processing(coupon_df)
    processed_vendor_coupon = vendor_coupon_processing(vendor_df=processed_vendor, coupon_df=processed_coupon,
                                                       stopwords_path=stopwords_path)

    tfidf_matrix = tfidf(processed_vendor_coupon.to_numpy())
    vendor_cosine_sim = cosine_similarity(tfidf_matrix)
    pickle.dump(vendor_cosine_sim, open(os.path.join(model_path, 'vendor_cosine_sim.pickle'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    vendor_id_arr = processed_vendor_coupon.index.to_numpy()
    pickle.dump(vendor_id_arr, open(os.path.join(model_path, 'vendor_id.pickle'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    rating_df.to_feather(os.path.join(model_path, 'rating_df.feather'))


def neural_net_evaluation(model, test_df):
    actual = test_df.rating.to_numpy()
    predictions = model.predict([test_df.user_id, test_df.gender, test_df.vendor_id, test_df.vd_country_id])
    predictions = np.array([a[0] for a in predictions])
    for i in range(len(actual)):
        print("Actual rating: ", actual[i], "vs Prediction: ", predictions[i])

    score_mean = mean_squared_error(actual, predictions)
    score_absolute = mean_absolute_error(actual, predictions)
    return score_mean, score_absolute