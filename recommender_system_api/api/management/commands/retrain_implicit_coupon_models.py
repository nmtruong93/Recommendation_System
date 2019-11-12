from django.core.management.base import BaseCommand
from recommender_system_api.utils.implicit.load_and_retrain_implicit import retrain_implicit_model


class Command(BaseCommand):
    help = 'Retrain implicit coupon models'

    def handle(self, *args, **kwargs):
        retrain_implicit_model(vendor=False)