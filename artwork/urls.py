# artwork/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.artwork_list, name='artwork_list'),
]