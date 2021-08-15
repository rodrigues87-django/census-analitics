import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from census.models import Census
from machine_learning_model.models import MachineLearningModel


def abrir_census(metodo):
    machine_learning_model = MachineLearningModel()
    machine_learning_model.open_classificador(metodo)
    return machine_learning_model


def treinar_arvores_de_decisao_census():
    machine_learning_model = get_census_data()
    machine_learning_model.treinar_arvores_decisao()
    salvar_modelo(machine_learning_model)
    print("treinar_arvores_de_decisao_census")
    return machine_learning_model


def treinar_regressao_logistica_census():
    machine_learning_model = get_census_data()
    machine_learning_model.treinar_regressao_logistica()
    salvar_modelo(machine_learning_model)
    print("treinar_regressao_logistica_census")

    return machine_learning_model


def treinar_knn_census():
    machine_learning_model = get_census_data()
    machine_learning_model.treinar_knn()
    salvar_modelo(machine_learning_model)
    print("treinar_knn_census")

    return machine_learning_model


def treinar_naive_bayes_census():
    machine_learning_model = get_census_data()
    machine_learning_model.treinar_naive_bayes()
    salvar_modelo(machine_learning_model)
    print("treinar_naive_bayes_census")

    return machine_learning_model


def treinar_random_forest_census():
    machine_learning_model = get_census_data()
    machine_learning_model.treinar_random_forest()
    salvar_modelo(machine_learning_model)
    print("treinar_random_forest_census")

    return machine_learning_model


def treinar_svm_census():
    machine_learning_model = get_census_data()
    machine_learning_model.treinar_svm()
    salvar_modelo(machine_learning_model)
    print("treinar_svm_census")

    return machine_learning_model


def salvar_modelo(machine_learning_model):
    machine_learning_model.save_classificador()
    machine_learning_model.definir_previsao(None)
    machine_learning_model.definir_precisao()
    machine_learning_model.save()


def get_census_data():
    machine_learning_model = MachineLearningModel()

    base = pd.DataFrame(list(Census.objects.all().values()))

    classe = base.iloc[:, 15].values
    labelencoder_classe = LabelEncoder()
    machine_learning_model.classe = labelencoder_classe.fit_transform(classe)

    previsores = base.iloc[:, 1:15].values

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

    previsores = onehotencoder.fit_transform(previsores).toarray()

    scaler = StandardScaler()
    machine_learning_model.previsores = scaler.fit_transform(previsores)

    machine_learning_model.previsores_treinamento, machine_learning_model.previsores_teste, machine_learning_model.classe_treinamento, machine_learning_model.classe_teste = \
        train_test_split(
            machine_learning_model.previsores,
            machine_learning_model.classe,
            test_size=0.15,
            random_state=0)

    return machine_learning_model
