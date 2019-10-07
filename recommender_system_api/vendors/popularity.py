from ..utils import queries
from urllib.request import urlopen
import json
from ..utils.connections import get_data_by_cursor, get_all_data_by_cursor
from django.conf import settings


# TODO: get vendor based on distance, rating

def popularity(user_id, balance, lat, long):
    public_key = get_data_by_cursor(query=queries.GET_USER_PUBLIC_KEY.format(user_id))[0]
    url = settings.DOMAIN_URL + "/get_balance/?public_key={}".format(public_key)
    balance = 0
    try:
        response = json.loads(urlopen(url=url).read())
        balance = response['result']['balance']
    except Exception as e:
        pass

    vendor_list = get_all_data_by_cursor(query=queries.GET_UPPER_BOUND_VENDORS.format(balance))
    if len(vendor_list) < queries.NUMBER_OF_RECOMMENDATIONS:
        vendor_list.append(
            get_all_data_by_cursor(query=queries.GET_LOWER_BOUND_VENDORS.format(balance)))
    vendor_list = [i[0] for i in vendor_list]

    return vendor_list
