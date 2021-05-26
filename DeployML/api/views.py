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
import uuid

from .apps import ApiConfig


class call_model(APIView):

    def post(self, request):
        if request.method == 'POST':
            """ video = request.FILES['video']
            if os.path.exists("1.mp4"):
                os.remove("1.mp4")
            path = default_storage.save('1.mp4', ContentFile(video.read()))


            ApiConfig.vsr.superVideo()
            print('done')"""
            userId = request.data['userId']
            uid = uuid.uuid4()
            if os.path.exists("static/"+userId) is False:
                os.mkdir("static/"+userId)
            file = open("static/"+userId+"/ids.txt", "a")
            file.write("{}\n".format(str(uid)))
            file.close()
            shutil.copy2('results/final_video.mp4', "static/"+userId+"/"+str(uid)+".mp4")
            shutil.copy2('results/output/00000000.png', "static/"+userId+"/"+str(uid)+".png")

            return HttpResponse(status=200)