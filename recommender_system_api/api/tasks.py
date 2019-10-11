from .views import update_model
import logging
from django_cron import CronJobBase, Schedule

# Get an instance of a logger
logger = logging.getLogger(__name__)


class LoadModelsCronJob(CronJobBase):
    """
        LoadModelsCronJob
    """
    RUN_EVERY_MINS = 30

    schedule = Schedule(run_every_mins=RUN_EVERY_MINS)
    code = 'load_models_cron_job'  # a unique code

    def do(self):
        update_model()
