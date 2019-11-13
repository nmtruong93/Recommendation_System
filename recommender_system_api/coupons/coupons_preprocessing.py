import warnings
import pandas as pd
from recommender_system_api.utils.explicit.words_processing import words_processing
warnings.filterwarnings('ignore')


def coupon_processing(coupon_df, stopwords_path):
    """
    Text processing related to coupon
    :param coupon_df: coupon dataframe from database
    :param stopwords_path: path of stopwords files
    :return: Series with index: coupon_id and column: bag_of_words
    """
    coupon_df.fillna(value='', inplace=True)

    coupon = coupon_df[['coupon_id', 'coupon_name', 'coupon_description']].drop_duplicates().groupby('coupon_id') \
        .agg(' '.join)

    cp_search_tags = coupon_df[['coupon_id', 'coupon_searchtags']].drop_duplicates().groupby('coupon_id') \
        .agg(' '.join)

    vendor = coupon_df[['coupon_id', 'vendor_name', 'vendor_des']].drop_duplicates().groupby('coupon_id') \
        .agg(' '.join)

    category_des = coupon_df[['coupon_id', 'vendor_cate_name', 'vendor_cate_des']].drop_duplicates() \
        .groupby('coupon_id').agg(' '.join)

    vd_search_tags = coupon_df[['coupon_id', 'vendor_searchtags']].drop_duplicates().groupby('coupon_id').agg(' '.join)

    vd_address = coupon_df[['coupon_id', 'address']].drop_duplicates().groupby('coupon_id').agg(' '.join)

    join_df = [coupon, cp_search_tags, vendor, category_des, vd_search_tags, vd_address]
    coupon_df = pd.concat(join_df, axis=1, join='inner')

    coupon = words_processing(coupon_df, stopwords_path=stopwords_path)

    return coupon
