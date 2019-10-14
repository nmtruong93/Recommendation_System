import pandas as pd
from config.settings import base
import os
import pickle
from recommender_system_api.coupons.coupons_preprocessing import coupon_processing
from recommender_system_api.utils.connections import get_data_by_pandas
from recommender_system_api.utils.words_processing import tfidf, cosine_similar
from recommender_system_api.utils import queries


def cb_coupon_recommendations(coupon_id, coupon_indices_list, cosine_sim):
    """
    Recommend 10 coupons with highest cosine score
    :param coupon_indices_list: list of coupon_id
    :param coupon_id: coupon id that user clicked
    :param cosine_sim: cosine similarity
    :return:
    """
    if coupon_id not in coupon_indices_list:
        #TODO: List popular coupon
        coupon_recommendations = []
    else:
        coupon_indices = pd.Series(coupon_indices_list)

        idx = coupon_indices[coupon_indices == coupon_id].index[0]

        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
        score_indices = score_series.index

        coupon_recommendations = [coupon_indices_list[i] for i in score_indices]

    return coupon_recommendations


def load_coupon_models():

    model_path = os.path.join(base.BASE_DIR, 'recommender_system_api/models/')

    cosine_similarity = pickle.load(open(os.path.join(model_path, 'cosine_similarity.pickle'), 'rb'))
    coupon_indices = pickle.load(open(os.path.join(model_path, 'coupon_indices.pickle'), 'rb'))

    return cosine_similarity, coupon_indices


def retrain_coupon_models():
    stopwords_path = os.path.join(base.BASE_DIR, 'recommender_system_api/utils/')
    model_path = os.path.join(base.BASE_DIR, 'recommender_system_api/models/')
    coupon_df = get_data_by_pandas(query=queries.GET_COUPON)
    coupon = coupon_processing(coupon_df, stopwords_path=stopwords_path)
    tfidf_vector = tfidf(coupon.to_numpy())

    coupon_indices = coupon.index.to_numpy()
    cosine_similarity = cosine_similar(tfidf_vector)

    pickle.dump(coupon_indices, open(os.path.join(model_path, 'coupon_indices.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(cosine_similarity, open(os.path.join(model_path, 'cosine_similarity.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("="*150)
    print("COUPON MODEL RETRAINED")
    print("=" * 150)