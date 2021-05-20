from django.apps import AppConfig
import sys
sys.path.insert(1, 'C:/No C/Work/Out/Video-Super-Resolution')


from super_video import SuperVideo


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    vsr = SuperVideo()