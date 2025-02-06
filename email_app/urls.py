from django.urls import path
from . import views

urlpatterns = [
    path('send/', views.send_email, name='send_email'),
    path('email_list/', views.email_list, name='email_list'),
    path('email_detail/<int:email_id>/', views.email_detail, name='email_detail'),
]