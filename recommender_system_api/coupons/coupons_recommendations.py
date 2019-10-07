import pandas as pd


def content_based_recommendations(id, coupon_indices_list, cosine_sim):
    """
    Recommend 10 coupons with highest cosine score
    :param coupon_indices_list: list of coupon_id
    :param id: coupon id that user clicked
    :param cosine_sim: cosine similarity
    :return:
    """
    coupon_indices = pd.Series(coupon_indices_list)

    idx = coupon_indices[coupon_indices == id].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    score_indices = score_series.index

    coupon_recommendations = [coupon_indices_list[i] for i in score_indices]

    return coupon_recommendations
