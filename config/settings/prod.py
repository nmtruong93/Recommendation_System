from .base import *

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1'
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'recommender_system_api',
        'USER': 'postgres',
        'PASSWORD': 'JaziBa73ac',
        'HOST': 'teecoin-backend-prod.cnnf42vc9hvq.ap-southeast-1.rds.amazonaws.com',
        'PORT': '5432',
    },
    'remote': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'tcoin_api',
        'USER': 'reportuser',
        'PASSWORD': 'TeeCoin@2018!@#',
        'HOST': 'teecoin-backend-prod.cnnf42vc9hvq.ap-southeast-1.rds.amazonaws.com',
        'PORT': '5432',
    },
}

DOMAIN_URL = 'https://api.tee-coin.com/v4'
