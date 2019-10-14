"""
WSGI config for config project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os
import sys
import site

site.addsitedir('/home/ubuntu/.virtualenv/recommender_system_api/lib/python3.6/site-packages')

# Add the app's directory to the PYTHONPATH
sys.path.append('/home/ubuntu/projects/recommender_system_api')
sys.path.append('/home/ubuntu/projects/recommender_system_api/recommender_system_api')

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.prod")

application = get_wsgi_application()