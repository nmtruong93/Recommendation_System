from ..utils.words_processing import words_processing
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def vendor_processing(vendor_df):

    vendor_df.fillna(value='', inplace=True)
    vendor_1 = vendor_df[['vendor_id', 'vendor_name', 'vendor_des']].drop_duplicates().groupby('vendor_id')\
        .agg(' '.join)

    vendor_2 = vendor_df[['vendor_id', 'vendor_cate_name', 'vendor_cate_des']].drop_duplicates().groupby('vendor_id')\
        .agg(' '.join)

    vendor_3 = vendor_df[['vendor_id', 'vendor_searchtags']].drop_duplicates().groupby('vendor_id')\
        .agg(' '.join)

    vendor_4 = vendor_df[['vendor_id', 'address']].drop_duplicates().groupby('vendor_id') \
        .agg(' '.join)

    join_df = [vendor_1, vendor_2, vendor_3, vendor_4]
    vendor_df = pd.concat(join_df, axis=1, join='inner')

    return vendor_df


def coupon_processing(coupon_df):

    coupon_df.fillna(value='', inplace=True)
    coupon_1 = coupon_df[['vendor_id', 'coupon_name', 'coupon_description']].drop_duplicates().groupby('vendor_id')\
        .agg(' '.join)

    coupon_2 = coupon_df[['vendor_id', 'coupon_searchtags']].drop_duplicates().groupby('vendor_id')\
        .agg(' '.join)

    join_df = [coupon_1, coupon_2]
    coupon_df = pd.concat(join_df, axis=1, join='inner')
    return coupon_df


def vendor_coupon_processing(vendor_df, coupon_df, stopwords_path):
    """
    Word processing for vendor and coupon
    :param stopwords_path: path of stopwords file
    :param vendor_df: DataFrame - output from vendor_processing function
    :param coupon_df: DataFrame - output from coupon_processing function
    :return: vendor_df: DataFrame - used as input to calculate cosine similarity and build_user_profile function
    """
    vendor_coupon = pd.concat([vendor_df, coupon_df], axis=1, join='inner')
    vendor = words_processing(vendor_coupon, stopwords_path=stopwords_path)
    return vendor











