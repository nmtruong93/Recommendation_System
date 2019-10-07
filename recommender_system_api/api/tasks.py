from .views import update_model
from celery import task
import logging

# Get an instance of a logger
logger = logging.getLogger(__name__)


@task()
def update_models():
    update_model()
