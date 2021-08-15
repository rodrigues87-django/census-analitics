from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from django.db.models import *
import pickle

from machine_learning_model.constants import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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

    def treinar_regressao_logistica(self):
        self.biblioteca_usada = REGRECAO_LOGISTICA
        self.classificador = LogisticRegression()
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

    def treinar_arvores_decisao(self):
        self.biblioteca_usada = ARVORES_DE_DECISAO
        self.classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

    def treinar_knn(self):
        self.biblioteca_usada = KNN
        self.classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

    def treinar_naive_bayes(self):
        self.biblioteca_usada = NAIVE_BAYES
        self.classificador = GaussianNB()
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

    def treinar_random_forest(self):
        self.biblioteca_usada = RANDOM_FOREST
        self.classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

    def treinar_svm(self):
        self.biblioteca_usada = SVM
        self.classificador = SVC(kernel='linear', random_state=1)
        self.classificador.fit(self.previsores_treinamento, self.classe_treinamento)

    def definir_previsao(self, previsores_teste):
        if previsores_teste is None:
            previsores_teste = self.previsores_teste
        self.previsao = self.classificador.predict(previsores_teste)

    def definir_precisao(self):
        self.precisao = accuracy_score(self.classe_teste, self.previsao)
        self.matriz = confusion_matrix(self.classe_teste, self.previsao)

    def save_classificador(self):
        with open(self.biblioteca_usada + '.pkl', 'wb') as f:
            pickle.dump(self.classificador, f)

    def open_classificador(self, metodo):
        try:
            with open(metodo + '.pkl', 'rb') as f:
                self.classificador = pickle.load(f)
        except Exception:
            self.classificador = None
