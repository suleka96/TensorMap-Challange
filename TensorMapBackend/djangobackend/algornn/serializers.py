from django.contrib.auth.models import User, Group
from rest_framework import serializers
from . models import RNNConfig,responseConfig


class ConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = RNNConfig
        fields = '__all__'

class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = responseConfig
        fields = '__all__'
