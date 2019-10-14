from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from ..vendors.vendor_hybrid_recommendation import recommended_for_you, specific_recommendation
from ..vendors.vendor_hybrid_recommendation import load_vendor_models, retrain_vendor_models
import logging
from ..coupons.coupons_recommendations import cb_coupon_recommendations, load_coupon_models, retrain_coupon_models
# Get an instance of a logger
logger = logging.getLogger(__name__)

# TODO: Set as key of REDIS,
vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model = load_vendor_models()
cosine_similarity, coupon_indices = load_coupon_models()

@api_view(['GET'])
@renderer_classes([JSONRenderer])
def get_recommended_for_you(request, user_id=8306, gender=1, new_user=False, has_reviewed=False):
    """

    :param request:
    :param user_id:
    :param gender:
    :param new_user:
    :param has_reviewed:
    :return:
    """
    recommendation = recommended_for_you(user_id, gender, vendor_id_arr, rating_df,
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


@api_view(['GET'])
@renderer_classes([JSONRenderer])
def get_coupons_for_you(request, coupon_id=16):
    """

    :param request:
    :param coupon_id:
    :return:
    """
    recommendation = cb_coupon_recommendations(coupon_id, coupon_indices, cosine_similarity)

    return Response({'coupons_for_you': recommendation})