# coding=utf-8
from django.urls import path
from . import views

urlpatterns = [
    path('recommended_for_you/', views.get_recommended_for_you,
         name='recommended_for_you'),

    path('custom/', views.get_custom_page_recommendations,
         name='get_custom_recommendation'),

    path('coupons_for_you/', views.get_coupons_for_you,
         name='coupons_for_you')
]
