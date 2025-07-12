from django.urls import path
from . import views

urlpatterns = [
    path('', views.cpx_widget, name='cpx_widget'),
] 