import joblib
import numpy as np
from ndarraydjango.fields import NDArrayField
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from django.db.models import *
import pickle

from census.models import Census
from machine_learning_model.constants import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from census.models import Census


class MachineLearningModel(Model):
    classe = None
    previsores = None
    previsores_treinamento = None
    previsores_teste = None
    classe_teste = None
    classe_treinamento = None
    classificador = None
    previsao = None
    matriz = None
    precisao = DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    biblioteca_usada = CharField(max_length=20)
    datetime = DateTimeField(auto_now_add=True)
    classe_modelo = None

    def __init__(self, biblioteca_usada, classe_modelo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classe_modelo = classe_modelo
        self.biblioteca_usada = biblioteca_usada

        self.open_classificador()
        self.definir_precisao()


    def open_classificador(self):
        try:
            with open(self.biblioteca_usada + '.pkl', 'rb') as f:
                self.classificador = pickle.load(f)
        except Exception as e:
            print(e)
            self.treinar_modelo()

    def save_classificador(self):
        with open(self.biblioteca_usada + '.pkl', 'wb') as f:
            pickle.dump(self.classificador, f)

    def treinar_modelo(self):
        query_data = self.classe_modelo.objects.all()
        self.get_data(query_data)
        self.dividir_base_previsores_classe()

        if self.biblioteca_usada is REGRESSAO_LOGISTICA:
            self.treinar_regressao_logistica()
        if self.biblioteca_usada is ARVORES_DE_DECISAO:
            self.treinar_arvores_decisao()
        if self.biblioteca_usada is KNN:
            self.treinar_svm()
        if self.biblioteca_usada is NAIVE_BAYES:
            self.treinar_naive_bayes()
        if self.biblioteca_usada is RANDOM_FOREST:
            self.treinar_random_forest()
        if self.biblioteca_usada is SVM:
            self.treinar_svm()
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

        self.save_classificador()

    def treinar_regressao_logistica(self):
        self.classificador = LogisticRegression(max_iter=10)

    def treinar_arvores_decisao(self):
        self.classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)

    def treinar_knn(self):
        self.classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    def treinar_naive_bayes(self):
        self.classificador = GaussianNB()

    def treinar_random_forest(self):
        self.classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

    def treinar_svm(self):
        self.classificador = SVC(kernel='linear', random_state=1)

    def get_data(self, query_data):

        database = pd.DataFrame(list(query_data.values()))

        if self.classe_modelo is Census:
            self.get_census_data(database)

    def get_census_data(self, database):

        classe = database.iloc[:, 15].values
        labelencoder_classe = LabelEncoder()
        self.classe = labelencoder_classe.fit_transform(classe)

        previsores = database.iloc[:, 1:15].values

        labelencoder_previsores = LabelEncoder()
        previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
        previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
        previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
        previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
        previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
        previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
        previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
        previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])
        onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
                                          remainder='passthrough')

        myscaler_name = "scaler_" + str(self.biblioteca_usada) + "_.bin"
        onehotencoder_name = "onehot_" + str(self.biblioteca_usada) + "_.joblib"

        if len(previsores) > 1:

            onehot = onehotencoder.fit(previsores)
            previsores = onehot.transform(previsores).toarray()

            scaler = StandardScaler()
            self.previsores = scaler.fit_transform(previsores)

            joblib.dump(onehot, onehotencoder_name)
            joblib.dump(scaler, myscaler_name)

        else:
            onehot = joblib.load(onehotencoder_name)
            scaler = joblib.load(myscaler_name)

            previsores = onehot.transform(previsores).toarray()
            self.previsores = scaler.fit_transform(previsores)

        print("termino codificacao")

    def dividir_base_previsores_classe(self):
        self.previsores_treinamento, self.previsores_teste, self.classe_treinamento, self.classe_teste = \
            train_test_split(
                self.previsores,
                self.classe,
                test_size=0.15,
                random_state=0)

        print("termino divisao")

    def definir_precisao(self):
        if self.previsores_teste is None:
            self.previsores_teste = self.previsores

        self.previsao = self.classificador.predict(self.previsores_teste)

        if self.classe_teste is not None:
            self.precisao = accuracy_score(self.classe_teste, self.previsao)
            self.matriz = confusion_matrix(self.classe_teste, self.previsao)
