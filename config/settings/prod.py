from .base import *

sys.path.insert(0, os.path.join(ROOT_DIR(), 'recommender_system_api'))

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    'recsys.tee-coin.com'
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
        'USER': 'postgres',
        'PASSWORD': 'JaziBa73ac',
        'HOST': 'teecoin-backend-prod.cnnf42vc9hvq.ap-southeast-1.rds.amazonaws.com',
        'PORT': '5432',
    },
}

DOMAIN_URL = 'https://api.tee-coin.com/v4'
WSGI_APPLICATION = 'config.wsgi_dev.application'
