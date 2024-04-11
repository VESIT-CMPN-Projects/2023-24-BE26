from django.urls import path
from . import views

urlpatterns = [
   path('',views.getRoutes,name="routes"),
   path('city/<str:city>', views.city, name='city'),
   path('city_bar/<str:city>', views.citywise_bar, name='city_bar'),
   path('city_pie/', views.citywise_poweroutputs, name='all_powers'),
  
]