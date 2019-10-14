from .base import *

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    'recsys-dev.tee-coin.com'
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'recommender_system_api',
        'USER': 'postgres',
        'PASSWORD': 'Nar85izwey',
        'HOST': 'teecoin-backend-test.cnnf42vc9hvq.ap-southeast-1.rds.amazonaws.com',
        'PORT': '5432',
    },
    'remote': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'tcoin_api_staging',
        'USER': 'postgres',
        'PASSWORD': 'Nar85izwey',
        'HOST': 'teecoin-backend-test.cnnf42vc9hvq.ap-southeast-1.rds.amazonaws.com',
        'PORT': '5432',
    },
}

DOMAIN_URL = 'https://api-dev.tee-coin.com/v4'

WSGI_APPLICATION = 'config.wsgi_dev.application'
