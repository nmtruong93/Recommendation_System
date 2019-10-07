# coding=utf-8
from django.urls import path
from . import views

urlpatterns = [
    path('main/<int:user_id>/<int:gender>', views.get_main_page_recommendations,
         name='get_main_recommendation'),

    path('custom/', views.get_custom_page_recommendations,
         name='get_custom_recommendation')
]