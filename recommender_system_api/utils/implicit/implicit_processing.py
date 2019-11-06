from recommender_system_api.utils.implicit.elasticsearch_queries import get_active_vendors, get_vendor_detail_views, \
    get_coupon_detail_views
from elasticsearch import Elasticsearch
import pandas as pd

client = Elasticsearch(hosts='https://search-teecoin-api-2xmwd6wn4nrdew7euzgaowljpm.ap-southeast-1.es.amazonaws.com/', use_ssl=True, verify_certs=False, timeout=60)


def get_item_detail_views_data(vendor=True, vendor_index=3, coupon_index=4):
    active_item_responses = client.search(index='catalogue-coupons', body=get_active_vendors())
    if vendor:
        active_item_ids = [i['_source']['vendor']['id'] for i in active_item_responses['hits']['hits']]
        detail_view_responses = client.search(index='access-logs', body=get_vendor_detail_views(active_item_ids))['hits']['hits']
    else:
        active_item_ids = [i['_source']['id'] for i in active_item_responses['hits']['hits']]
        detail_view_responses = client.search(index='access-logs', body=get_coupon_detail_views(active_item_ids))['hits']['hits']

    item_id_list, account_id_list, rating_list = [], [], []
    for view in detail_view_responses:
        if vendor:
            item_id = int(view['_source']['path'].split('/')[vendor_index])
        else:
            item_id = int(view['_source']['path'].split('/')[coupon_index])
        account_id = view['_source']['account']
        item_id_list.append(item_id)
        account_id_list.append(account_id)
        rating_list.append(4)
    implicit_df = pd.DataFrame({'actual_account_id': account_id_list, 'actual_item_id': item_id_list, 'rating': rating_list})
    implicit_df.dropna(inplace=True)
    implicit_df.drop_duplicates(inplace=True)
    implicit_df['actual_account_id'] = implicit_df.actual_account_id.astype('int64')
    return implicit_df
