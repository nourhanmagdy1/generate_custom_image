from django.urls import path
from . import views

urlpatterns = [
    path('', views.generate_image_interface, name='generate_image_interface'),
    path('generate-image/', views.generate_image, name='generate_image'),
]
