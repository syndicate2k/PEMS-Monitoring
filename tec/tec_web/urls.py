from django.urls import path
from .views import home_page, tube_1_page, tube_2_page, tube_3_page, graphic_page

urlpatterns = [
    path('', home_page, name='home'),
    path('tube/1', tube_1_page, name='tube1'),
    path('tube/2', tube_2_page, name='tube2'),
    path('tube/3', tube_3_page, name='tube3'),
    path('graphic/', graphic_page, name='graphic'),
]