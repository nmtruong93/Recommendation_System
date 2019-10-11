from django.core.management.base import BaseCommand
from recommender_system_api.api.views import update_model


class Command(BaseCommand):
    help = 'Create random users'

    def handle(self, *args, **kwargs):
      update_model()