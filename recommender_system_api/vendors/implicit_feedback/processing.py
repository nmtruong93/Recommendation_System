from recommender_system_api.utils.elasticsearch_queries import get_active_vendors, get_vendor_detail_views, \
    get_coupon_detail_views
from elasticsearch import Elasticsearch
import pandas as pd

client = Elasticsearch(hosts='https://search-teecoin-api-2xmwd6wn4nrdew7euzgaowljpm.ap-southeast-1.es.amazonaws.com/', use_ssl=True, verify_certs=False, timeout=60)


def get_vendor_detail_views_data():
    active_vendor_responses = client.search(index='catalogue-coupons', body=get_active_vendors())
    vendor_ids = [i['_source']['vendor']['id'] for i in active_vendor_responses['hits']['hits']]

    detail_view_responses = client.search(index='access-logs', body=get_vendor_detail_views(vendor_ids))['hits']['hits']
    vendor_id_list, account_id_list, rating_list = [], [], []
    for view in detail_view_responses:
        vendor_id = int(view['_source']['path'].split('/')[3])
        account_id = view['_source']['account']
        vendor_id_list.append(vendor_id)
        account_id_list.append(account_id)
        rating_list.append(4)
    implicit_df = pd.DataFrame({'actual_account_id': account_id_list, 'actual_vendor_id': vendor_id_list, 'rating': rating_list})
    implicit_df.dropna(inplace=True)
    implicit_df.drop_duplicates(inplace=True)
    implicit_df['actual_account_id'] = implicit_df.actual_account_id.astype('int64')
    return implicit_df


def get_coupon_detail_views_data():

    response = client.search(index="access-logs", body=get_coupon_detail_views())

    return response


if __name__ == '__main__':
    re = get_coupon_detail_views_data()
    print(re)