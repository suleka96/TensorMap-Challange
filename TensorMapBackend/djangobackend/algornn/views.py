from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
from . models import RNNConfig, responseConfig
from . serializers import ConfigSerializer,ResultSerializer
from django.views.decorators.csrf import csrf_exempt
# import create_graph
import pandas as pd
import tensorflow as tf
import numpy as np
import news20
import stockPred


@csrf_exempt
def RNNConfigView(request):

    if request.method == 'POST':
        netInfo = JSONParser().parse(request)
        # netInfo_serialized = ConfigSerializer(netInfo)

        if netInfo['dataset'] == "Stock Price":
           resObj = stockPred.RunModelStock(netInfo)
        
        elif netInfo['dataset'] == "20 News Groups":
          resObj =  news20.RunModel20News(netInfo)        
        
        res_serialized = ResultSerializer(resObj)        
        # metric_info = create_graph.runGraph(netInfo)
        return  JsonResponse(res_serialized.data, safe=False)
        # return HttpResponse(netInfo["rnnNodes"])