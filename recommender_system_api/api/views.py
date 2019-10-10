from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from ..vendors.vendor_hybrid_recommendation import main_page_recommendation, specific_recommendation
from ..vendors.vendor_hybrid_recommendation import processing_ouput, get_and_process_data
import logging

# Get an instance of a logger
logger = logging.getLogger(__name__)

# TODO: Set as key of REDIS,
vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model = processing_ouput()

@api_view(['GET', 'POST'])
@renderer_classes([JSONRenderer])
def get_main_page_recommendations(request, user_id=7973, gender=2, new_user=False, has_reviewed=False):
    """

    :param request:
    :param user_id:
    :param gender:
    :param new_user:
    :param has_reviewed:
    :return:
    """
    recommendation = main_page_recommendation(user_id, gender, vendor_id_arr, rating_df,
                                              vendor_cosine_sim, neural_net_model, new_user, has_reviewed)

    return Response({'vendor_recommendation': recommendation})


@api_view(['GET', 'POST'])
@renderer_classes([JSONRenderer])
def get_custom_page_recommendations(request, user_id=7973, gender=2, vendor_id=56507):
    """

    :param request:
    :param user_id:
    :param gender:
    :param vendor_id:
    :return:
    """
    recommendation = specific_recommendation(user_id, gender, vendor_id,
                                             vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model)

    return Response({'vendor_recommendation': recommendation})


def update_model():
    global vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model
    try:
        vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model = get_and_process_data()
    except Exception as error:
        print(str(error))
        logger.error('An error occurred (update_model): %s' % str(error), exc_info=True, extra={})
