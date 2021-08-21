import django
import numpy as np

django.setup()

from previsoes.models import Previsao

from unittest import TestCase

from census.models import Census
from machine_learning_model.constants import REGRESSAO_LOGISTICA
from machine_learning_model.models import MachineLearningModel
import pandas as pd


class TestMachineLearningModel(TestCase):

    def test_prever_regressao_logistica(self):
        machine_learning_model = MachineLearningModel()
        machine_learning_model.open_classificador(REGRESSAO_LOGISTICA, Census)
        previsao = Previsao()
        previsao.age = 40
        previsao.workclass = "State-gov"
        previsao.final_weight = 234721
        previsao.education = "Bachelors"
        previsao.education_num = 14
        previsao.marital_status = "Married-civ-spouse"
        previsao.occupation = "Exec-managerial"
        previsao.relationship = "Husband"
        previsao.race = "White"
        previsao.sex = "Female"
        previsao.capital_gain = 0
        previsao.capital_loos = 0
        previsao.hour_per_week = 45
        previsao.native_country = "United-States"
        previsao.save()

        query = Previsao.objects.filter(id=previsao.id)
        machine_learning_model.get_data(query)
        machine_learning_model.definir_precisao()

        print("termino")
