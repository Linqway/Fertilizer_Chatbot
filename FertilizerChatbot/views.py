from django.http import JsonResponse
import json

from rest_framework.decorators import api_view, permission_classes, authentication_classes

from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated

from rest_framework.response import Response
from rest_framework import status

from django.shortcuts import render


def index(req):
    return render(req,'index.html')

