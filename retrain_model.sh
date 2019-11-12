#!/bin/bash
cd /Users/teecoin/PycharmProjects/recommender_system_api
source ../recommender_env/bin/activate
python manage_dev.py retrain_vendor_models
python manage_dev.py retrain_coupon_models
python manage_dev.py retrain_implicit_vendor_models
python manage_dev.py retrain_implicit_coupon_models