from django.core.management.base import BaseCommand
from recommender_system_api.vendors.load_and_retrain import retrain_vendor_models


class Command(BaseCommand):
    help = 'Retrain vendor models'

    def handle(self, *args, **kwargs):
        retrain_vendor_models()