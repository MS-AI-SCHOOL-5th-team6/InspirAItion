"""team6 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from team6 import views
from app.views import index, post_detail, create_post, edit_post, delete_post  # app 앱의 views 모듈에서 뷰 함수를 임포트
from accounts.views import signup, profile_update  # accounts 앱의 views 모듈에서 뷰 함수를 임포트
from email_app.views import email_detail,email_list,send_email

urlpatterns = [
    path("admin/", admin.site.urls), # 관리자 페이지
    path("custom-admin/", views.custom_admin, name='custom_admin'), # 커스텀 관리자 페이지 경로 설정
    path('', views.home, name='home'),  # 홈 페이지 경로 설정
    path('about/', views.about, name='about'),  # About 페이지 경로 설정
    path('services/', views.services, name='services'),  # Services 페이지 경로 설정
    path('our_team/', views.our_team, name='our_team'),  # Our Team 페이지 경로 설정
    path('board/', views.board, name='board'),  # Board 페이지 경로 설정
    path('contact_us/', views.contact_us, name='contact_us'),  # Contact Us 페이지 경로 설정
    path('ai_play/', views.ai_play, name='ai_play'),
    path('art_gal/', views.art_gal, name='art_gal'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),  # 로그아웃 후 홈 페이지로 이동
    path('signup/', signup, name='signup'),
    path('profile/update/', profile_update, name='profile_update'),

    path('app/', include('app.urls')),
    path('api/', include('ai_playground.urls')),  # ai_playground 앱의 URL을 포함
    path('accounts/', include("accounts.urls")),
    path('ai_playground/', include('ai_playground.urls')),
    path('artwork/', include('artwork.urls')),  # artwork 앱의 URL을 
    path('email/', include('email_app.urls')), # 이메일 앱의 URL을 포함
    path('index/', index, name='index'),  # index 뷰를 다른 URL 경로와 연결
    path('post/<int:pk>/', post_detail, name='post_detail'),  # 포스트 상세 페이지 경로 설정
    path('post/create/', create_post, name='create_post'),  # 포스트 생성 페이지 경로 설정
    path('post/edit/<int:pk>/', edit_post, name='edit_post'),  # 포스트 수정 페이지 경로 설정
    path('post/delete/<int:pk>/', delete_post, name='delete_post'),  # 포스트 삭제 페이지 경로 설정
    path('send/', views.send_email, name='send_email'),
    path('email_list/', views.email_list, name='email_list'),
    path('email_detail/<int:email_id>/', views.email_detail, name='email_detail'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
