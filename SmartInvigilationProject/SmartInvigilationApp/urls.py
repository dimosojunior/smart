from django.urls import path
from . import views

urlpatterns = [
    path('', views.homePage, name="homePage"),
    path('signin/', views.signin, name="signin"),
    path('logout/', views.logout, name='logout'),

   ## path('WebcamInvigilation/', views.WebcamInvigilation, name='WebcamInvigilation'),
    path('project2/', views.project2, name='project2'),
    path('record_video/', views.record_video, name='record_video'),

    path('starting_page/', views.starting_page, name='starting_page'),

]