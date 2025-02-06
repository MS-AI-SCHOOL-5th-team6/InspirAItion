from django.urls import path
from . import views
'''
urlpatterns = [
    path("", views.index, name="index"),
    path("posts/new/", views.create_post, name="create_post"),
    path("posts/<int:pk>/", views.post_detail, name="post_detail"),
    path("posts/<int:pk>/edit/", views.edit_post, name="edit_post"),
    path("posts/<int:pk>/delete/", views.delete_post, name="delete_post"),
    path("ai/generate/", views.generate_image, name="generate_image"),
    path("gallery/my/", views.my_gallery, name="my_gallery"),
    path("gallery/public/", views.public_gallery, name="public_gallery")
]
'''

urlpatterns = [
    #path("index", views.index, name="index"),
    path("new/", views.create_post, name="create_post"),
    path("<int:pk>/edit/", views.edit_post, name="edit_post"),
    path("<int:pk>/", views.post_detail, name="post_detail"),
    path("<int:pk>/delete/", views.delete_post, name="delete_post"),
    path("index_ai", views.index_ai, name="index_ai"),
    path("ai/generate/", views.generate_image, name="generate_image"),
]

