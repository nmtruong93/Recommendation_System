from django.core.management.base import BaseCommand
from recommender_system_api.coupons.coupons_recommendations import retrain_coupon_models


class Command(BaseCommand):
    help = 'Retrain coupon models'

    def handle(self, *args, **kwargs):
        retrain_coupon_models()
