from rest_framework.serializers import ModelSerializer

from machine_learning_model.models import MachineLearningModel


class MachineLearningModelSerializer(ModelSerializer):
    class Meta:
        model = MachineLearningModel
        fields = 'precisao', 'biblioteca_usada'
