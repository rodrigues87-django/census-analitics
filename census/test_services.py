import django

django.setup()

from unittest import TestCase
from census.services import *
from machine_learning_model.constants import *


class Test(TestCase):
    def test_treinar_census(self):
        treinar_arvores_de_decisao_census()
        treinar_regressao_logistica_census()
        treinar_knn_census()
        treinar_naive_bayes_census()
        treinar_random_forest_census()
        treinar_svm_census()
        pass

    def test_abrir_census(self):
        machine_learning_model = abrir_census(ARVORES_DE_DECISAO)
        machine_learning_model = abrir_census(REGRECAO_LOGISTICA)
        machine_learning_model = abrir_census(KNN)
        machine_learning_model = abrir_census(NAIVE_BAYES)
        machine_learning_model = abrir_census(RANDOM_FOREST)
        machine_learning_model = abrir_census(REGRAS)
        machine_learning_model = abrir_census(SVM)

        pass


