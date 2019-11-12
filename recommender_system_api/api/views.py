from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from recommender_system_api.vendors.vendor_hybrid_recommendation import recommended_for_you, specific_recommendation
from recommender_system_api.vendors.load_and_retrain import load_vendor_models
import logging
from ..coupons.coupons_recommendations import cb_coupon_recommendations, load_coupon_models
from recommender_system_api.utils.implicit.implicit_processing import get_full_data
from recommender_system_api.utils.implicit.load_and_retrain_implicit import retrain_implicit_model, load_models
from recommender_system_api.utils.implicit.implicit_user_profiles import triple_user_profiles


# prediction = triple_user_profiles(account_id=1665, gender=0, model=triplet_model, rating_df=implicit_df)
# Get an instance of a logger
logger = logging.getLogger(__name__)

implicit_vendor_df, triplet_model = load_models(vendor=True)
vendor_id_arr, rating_df, vendor_cosine_sim, neural_net_model = load_vendor_models()
cosine_similarity, coupon_indices = load_coupon_models()

@api_view(['GET'])
@renderer_classes([JSONRenderer])
def get_recommended_for_you(request):
    """

    :param request:
    :param user_id:
    :param gender:
    :param new_user:
    :param has_reviewed:
    :return:
    """
    recommendation = []
    try:
        account_id = int(request.GET.get("account_id", 15898))
        gender = int(request.GET.get("account_id", 0))

        recommendation = recommended_for_you(account_id, gender, vendor_id_arr, rating_df,
                                             vendor_cosine_sim, neural_net_model, triplet_model, implicit_vendor_df)
    except Exception as e:
        print(str(e))

    return Response({'recommended_for_you': recommendation})


@api_view(['GET', 'POST'])
@renderer_classes([JSONRenderer])
def get_custom_page_recommendations(request, user_id=15898, gender=0, vendor_id=56507):
    """

    :param request:
    :param user_id:
    :param gender:
    :param vendor_id:
    :return:
    """
    recommendation = specific_recommendation(user_id, gender, vendor_id, vendor_id_arr, rating_df, vendor_cosine_sim,
                                             neural_net_model, triplet_model, implicit_vendor_df)

    return Response({'vendor_recommendation': recommendation})


@api_view(['GET'])
@renderer_classes([JSONRenderer])
def get_coupons_for_you(request):
    """

    :param request:
    :param coupon_id:
    :return:
    """

    recommendation = []
    try:
        coupon_id = int(request.GET.get("coupon_id", 6))
        recommendation = cb_coupon_recommendations(coupon_id, coupon_indices, cosine_similarity)
    except Exception as e:
        print(str(e))

    return Response({'coupons_for_you': recommendation})