import django

django.setup()

from census.services import *
from machine_learning_model.constants import *

machine_learning_model = abrir_census(ARVORES_DE_DECISAO)

treinar_arvores_de_decisao_census()
treinar_regressao_logistica_census()
treinar_knn_census()
treinar_naive_bayes_census()
treinar_random_forest_census()
treinar_svm_census()
