from django.contrib import admin
from . import views
from django.urls import path, include
from rest_framework import routers
from django.contrib.staticfiles.urls import staticfiles_urlpatterns


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('aqi', include('aqiApp.urls')),
    path('heatwave', include('heatwaveApp.urls')),
    path('source-code', views.sourcecode, name='source-code'),
    path('features', views.features, name='features'),
    path('team', views.team, name='team'),
]
handler500='exceptions.views.error_500'

urlpatterns += staticfiles_urlpatterns()


# Credits: Harsh Anand (Github: anand-harsh)