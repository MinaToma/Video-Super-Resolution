import os

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from wsgiref.util import FileWrapper
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import shutil

from .apps import ApiConfig


class call_model(APIView):

    def post(self, request):
        if request.method == 'POST':
            video = request.FILES['video']
            if os.path.exists("1.mp4"):
                os.remove("1.mp4")
            path = default_storage.save('1.mp4', ContentFile(video.read()))
            ApiConfig.vsr.superVideo()
            print('done')
            file = FileWrapper(open('results/final_video.mp4', 'rb'))
            response = HttpResponse(file, content_type='video/mp4')
            response['Content-Disposition'] = 'attachment; filename=my_video.mp4'
            return response