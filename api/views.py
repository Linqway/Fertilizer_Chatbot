import os
import json
import re


from rest_framework.views import APIView
from rest_framework.response import Response

# from rest_framework.exceptions import PermissionDenied, APIException

from FertilizerChatbot.configurations.configurationhandler import CONFIG

from . import permissions
from .helper import preprocess
from .helper import form

from .helper import dictionary

from .helper import train
chatbotTrain=train.ChatbotTrain()

from .helper import predict
chatbotPredict=predict.ChatbotPredict()

import logging
logger = logging.getLogger(__name__)

from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

"""
API For Immediate training of records
"""
class Train(APIView):

    def get(self, req):
        
        language = req.GET.get('language')
        isTrained = chatbotTrain.main(language)

        # if isTrained:
        return Response({
            "success": True,
            "result": "Model Trained for language "+language,
        })
        # else:
        #     raise APIException()

"""
API for Prediction of chat
"""
class Predict(APIView):

    @method_decorator(cache_page(CONFIG['prediction']['cache_ttl']['sec']))
    def get(self, req):
        
        input_dict = {
            "sentence" : req.GET.get('inputSentence'),
            "language" :req.GET.get('language','eng'),
            "tag" : req.GET.get('tag', None)
        }
        result = chatbotPredict.main(input_dict)
        return Response(result)

"""
API for Prediction of chat for Web User
"""
class Form(APIView):

    def post(self, req):
        # try:
            print(req.data)
            data = {
                "temparature" : int(req.data['Temprature']),
                "soilType" : req.data['SoilType'],
                "cropType" : req.data['CropType'],
                "humidity" : int(req.data['Humidity']),
                "moisture" : int(req.data['Moisture']),
                "nitrogen" : int(req.data['Nitrogen']),
                "potassium" : int(req.data['Potassium']),
                "phosphorous" : int(req.data['Phosphorous']),   
            }
            result = form.getFertilizerType(data,req.data['language'])
            return Response(result)
        # except:
        #     return Response({ "message" : "Wrong Input. PLease Enter Correct input in Fields"})
