from ..utils.words_processing import words_processing
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class VendorContentBased(object):

    def __init__(self, vendor_df, coupon_df, rating_df):
        self.vendor_df = vendor_df
        self.coupon_df = coupon_df
        self.rating_df = rating_df

    def vendor_processing(self):

        self.vendor_df.fillna(value='', inplace=True)
        vendor_1 = self.vendor_df[['vendor_id', 'vendor_name', 'vendor_des']].drop_duplicates().groupby('vendor_id')\
            .agg(' '.join)

        vendor_2 = self.vendor_df[['vendor_id', 'vendor_cate_name', 'vendor_cate_des']].drop_duplicates().groupby('vendor_id')\
            .agg(' '.join)

        vendor_3 = self.vendor_df[['vendor_id', 'vendor_searchtags']].drop_duplicates().groupby('vendor_id')\
            .agg(' '.join)
        vendor_df = vendor_1.merge(vendor_2, on='vendor_id', how='inner').merge(vendor_3, on='vendor_id', how='inner')
        return vendor_df

    def coupon_processing(self):

        self.coupon_df.fillna(value='', inplace=True)
        coupon_1 = self.coupon_df[['vendor_id', 'coupon_name', 'coupon_description']].drop_duplicates().groupby('vendor_id')\
            .agg(' '.join)

        coupon_2 = self.coupon_df[['vendor_id', 'coupon_searchtags']].drop_duplicates().groupby('vendor_id')\
            .agg(' '.join)

        coupon_df = coupon_1.merge(coupon_2, on='vendor_id', how='inner')
        return coupon_df

    @staticmethod
    def vendor_coupon_processing(vendor_df, coupon_df, stopwords_path):
        """
        Word processing for vendor and coupon
        :param stopwords_path: path of stopwords file
        :param vendor_df: DataFrame - output from vendor_processing function
        :param coupon_df: DataFrame - output from coupon_processing function
        :return: vendor_df: DataFrame - used as input to calculate cosine similarity and build_user_profile function
        """
        vendor_coupon = pd.merge(vendor_df, coupon_df, how='left', on='vendor_id')
        vendor = words_processing(vendor_coupon, stopwords_path=stopwords_path)
        return vendor.reset_index()











