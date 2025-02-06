from django.contrib.auth import views as auth_views
from django.urls import path
from . import views
'''
urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="accounts/login.html"),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(next_page="/app/"), name="logout"),
    path('profile/update/', views.profile_update, name='profile_update')
]
'''

urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    #path("logout/", auth_views.LogoutView.as_view(next_page="/app/"), name="logout"),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),  # 로그아웃 후 홈 페이지로 이동
    path('profile/update/', views.profile_update, name='profile_update')
]

